#!/usr/bin/env python3
"""Train CNN QRS-finding detectors and derive Brugada by rule."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from ecg_few.cnn.data import (
    ECGImageDataset,
    PreprocessConfig,
    compute_preprocess_config,
    load_image_tensor,
    load_raw_image,
)
from ecg_few.cnn.domain_adaptation import (
    DomainClassifier,
    GradientReversal,
    copy_encoder_weights,
    coral_loss,
    cycle_batches,
    domain_adaptation_weight,
    make_unlabeled_loader,
    mmd_loss,
    pretrain_encoder_simclr,
    resnet_features,
    resnet_logits_from_features,
)
from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.findings import LABEL_NAMES, findings_to_brugada
from ecg_few.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    BrugadaImageRow,
    parse_int_list,
    patients_from_rows,
    read_jsonl,
    read_manifest,
    resolve_folds_path,
    rows_for_patient_ids,
    select_context_patient_ids,
    selection_for,
    validate_fold_plan,
)

DEFAULT_MODEL_NAME = "resnet18"
ONLY_LEAD = "V1"


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, 0].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CNN few-shot LOOCV for QRS findings and derived Brugada."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=Path(""),
        help="Optional synthetic QRS dataset used for detector training instead of fold context.",
    )
    parser.add_argument(
        "--external-train-mode",
        choices=("shared", "per_fold"),
        default="shared",
        help="When using --train-dataset-root, train once per k/seed or once per real fold.",
    )
    parser.add_argument("--folds", type=Path, default=Path(""))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cnn_loocv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/loocv/cnn"))
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-strategy",
        choices=("fixed", "val_derived_balanced_accuracy", "val_derived_f1"),
        default="val_derived_balanced_accuracy",
    )
    parser.add_argument("--loss-pos-weight", choices=("none", "auto"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resnet-weights", choices=("default", "none"), default="default")
    parser.add_argument(
        "--target-dataset-root",
        type=Path,
        default=Path(""),
        help=(
            "Optional unlabeled target-domain dataset for SSL/CORAL/MMD/DANN. "
            "Defaults to --dataset-root when --train-dataset-root is used."
        ),
    )
    parser.add_argument(
        "--domain-adaptation",
        choices=("none", "coral", "mmd", "dann"),
        default="none",
    )
    parser.add_argument("--domain-adaptation-weight", type=float, default=0.1)
    parser.add_argument(
        "--mmd-kernel-scales",
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated RBF kernel scales for MMD feature alignment.",
    )
    parser.add_argument("--ssl-pretrain-epochs", type=int, default=0)
    parser.add_argument("--ssl-pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--ssl-projection-dim", type=int, default=128)
    parser.add_argument("--ssl-temperature", type=float, default=0.2)
    parser.add_argument("--gradcam-count", type=int, default=6)
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    folds_path = resolve_folds_path(dataset_root, args.folds)
    output_root = Path(args.output_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    rows = only_v1_rows(read_manifest(dataset_root / "labels" / "all_labels.csv"))
    has_train_dataset = str(args.train_dataset_root) not in {"", "."}
    train_dataset_root = (
        Path(args.train_dataset_root).resolve() if has_train_dataset else dataset_root
    )
    train_pool_rows = (
        only_v1_rows(read_manifest(train_dataset_root / "labels" / "all_labels.csv"))
        if has_train_dataset
        else None
    )
    needs_target_domain = (
        str(args.domain_adaptation) != "none" or int(args.ssl_pretrain_epochs) > 0
    )
    target_dataset_root = resolve_target_dataset_root(
        args=args,
        dataset_root=dataset_root,
        has_train_dataset=has_train_dataset,
        needs_target_domain=needs_target_domain,
    )
    target_rows = (
        only_v1_rows(read_manifest(target_dataset_root / "labels" / "all_labels.csv"))
        if target_dataset_root is not None
        else None
    )
    folds = read_jsonl(folds_path.resolve())
    k_values = parse_int_list(str(args.k_values))
    seeds = parse_int_list(str(args.seeds))
    validate_fold_plan(folds, rows, k_values=k_values, seeds=seeds)
    if int(args.limit_folds) > 0:
        folds = folds[: int(args.limit_folds)]
    device = resolve_device(str(args.device))
    print(f"[INFO] CNN LOOCV device: {device}")

    summary_rows: list[dict[str, object]] = []
    model_name = DEFAULT_MODEL_NAME
    for k in k_values:
        for seed in seeds:
            run_dir = output_root / model_name / f"k{k}_seed{seed}"
            predictions = run_k_seed(
                rows=rows,
                train_pool_rows=train_pool_rows,
                folds=folds,
                dataset_root=dataset_root,
                train_dataset_root=train_dataset_root,
                target_dataset_root=target_dataset_root,
                target_rows=target_rows,
                run_dir=run_dir,
                k=k,
                seed=seed,
                args=args,
                device=device,
            )
            summary_rows.append(
                summarize_predictions(
                    predictions,
                    k=k,
                    seed=seed,
                    run_dir=run_dir,
                    report_dir=report_dir,
                    model_name=model_name,
                )
            )
    write_summary_reports(summary_rows, report_dir)


def only_v1_rows(rows: Sequence[BrugadaImageRow]) -> list[BrugadaImageRow]:
    return [row for row in rows if row.lead.upper() == ONLY_LEAD]


def run_k_seed(
    *,
    rows: Sequence[BrugadaImageRow],
    train_pool_rows: Sequence[BrugadaImageRow] | None,
    folds: Sequence[dict[str, Any]],
    dataset_root: Path,
    train_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    run_dir: Path,
    k: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    external_train_mode = str(getattr(args, "external_train_mode", "shared"))
    if train_pool_rows is not None and external_train_mode == "shared":
        return run_k_seed_shared_external_training(
            rows=rows,
            train_pool_rows=train_pool_rows,
            folds=folds,
            dataset_root=dataset_root,
            train_dataset_root=train_dataset_root,
            target_dataset_root=target_dataset_root,
            target_rows=target_rows,
            run_dir=run_dir,
            k=k,
            seed=seed,
            args=args,
            device=device,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "run_config.json",
        {
            "architecture": DEFAULT_MODEL_NAME,
            "k": k,
            "seed": seed,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "threshold_strategy": str(args.threshold_strategy),
            "loss_pos_weight": str(args.loss_pos_weight),
            "resnet_weights": str(args.resnet_weights),
            "domain_adaptation": str(args.domain_adaptation),
            "domain_adaptation_weight": float(args.domain_adaptation_weight),
            "ssl_pretrain_epochs": int(args.ssl_pretrain_epochs),
            "target_dataset_root": ""
            if target_dataset_root is None
            else target_dataset_root.as_posix(),
            "target_domain_policy": "fold_exclusive" if target_rows is not None else "none",
        },
    )
    output_path = run_dir / "fold_predictions.jsonl"
    resume = bool(getattr(args, "resume", True))
    existing = existing_keys(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    predictions: list[dict[str, object]] = load_existing_predictions(output_path)
    gradcam_written = 0
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for(fold, k=k, seed=seed)
        context_ids = selection["context_patient_ids"]
        validation_ids = selection["validation_patient_ids"]
        current_train_root = dataset_root
        if train_pool_rows is None:
            train_rows = rows_for_patient_ids(rows, context_ids)
            train_context_ids = context_ids
        else:
            train_context_ids = select_qrs_training_patient_ids(
                train_pool_rows,
                k=k,
                seed=seed + fold_id * 100_003,
            )
            train_rows = rows_for_patient_ids(train_pool_rows, train_context_ids)
            current_train_root = train_dataset_root
        val_rows = rows_for_patient_ids(rows, validation_ids)
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        print(
            f"[INFO] CNN k={k} seed={seed} fold={fold_id} "
            f"test_patient={test_patient_id}"
        )
        result = train_and_predict_fold(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            train_dataset_root=current_train_root,
            eval_dataset_root=dataset_root,
            target_dataset_root=target_dataset_root,
            target_rows=exclude_patient_rows(target_rows, test_patient_id),
            fold_dir=run_dir / f"fold_{fold_id:04d}",
            fold_id=fold_id,
            seed=seed,
            args=args,
            device=device,
            write_gradcam=gradcam_written < int(args.gradcam_count),
        )
        gradcam_written += int(result.pop("gradcam_written", 0))
        result.update(
            {
                "key": key,
                "fold_id": fold_id,
                "k": k,
                "seed": seed,
                "test_patient_id": test_patient_id,
                "context_patient_ids": "|".join(train_context_ids),
                "validation_patient_ids": "|".join(validation_ids),
            }
        )
        append_jsonl(output_path, result)
        predictions.append(result)
        write_csv(run_dir / "fold_predictions.csv", predictions)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def train_and_predict_fold(
    *,
    train_rows: Sequence[BrugadaImageRow],
    val_rows: Sequence[BrugadaImageRow],
    test_rows: Sequence[BrugadaImageRow],
    train_dataset_root: Path,
    eval_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    fold_dir: Path,
    fold_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    write_gradcam: bool,
) -> dict[str, object]:
    set_seed(seed + fold_id * 100_003)
    fold_dir.mkdir(parents=True, exist_ok=True)
    preprocess = compute_preprocess_config(
        train_rows,
        dataset_root=train_dataset_root,
        image_size=int(args.image_size),
    )
    train_loader = make_loader(
        train_rows,
        dataset_root=train_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
    )
    val_loader = (
        make_loader(
            val_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            seed=seed,
            device=device,
        )
        if val_rows and all(row.has_qrs_labels for row in val_rows)
        else None
    )
    model = build_model(resnet_weights=str(args.resnet_weights)).to(device)
    if target_dataset_root is not None and target_rows and int(args.ssl_pretrain_epochs) > 0:
        encoder, _ssl_payload = pretrain_encoder_simclr(
            target_rows=target_rows,
            target_dataset_root=target_dataset_root,
            preprocess=preprocess,
            run_dir=fold_dir / "ssl_pretrain",
            seed=seed + fold_id * 100_003,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            resnet_weights=str(args.resnet_weights),
            epochs=int(args.ssl_pretrain_epochs),
            lr=float(args.ssl_pretrain_lr),
            weight_decay=float(args.weight_decay),
            projection_dim=int(args.ssl_projection_dim),
            temperature=float(args.ssl_temperature),
        )
        if encoder is not None:
            copy_encoder_weights(model, encoder)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=loss_pos_weight(train_rows).to(device)
        if str(args.loss_pos_weight) == "auto"
        else None
    )
    domain_classifier = (
        DomainClassifier().to(device) if str(args.domain_adaptation) == "dann" else None
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + ([] if domain_classifier is None else list(domain_classifier.parameters())),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    target_loader = make_target_loader(
        target_rows=target_rows,
        target_dataset_root=target_dataset_root,
        preprocess=preprocess,
        args=args,
        seed=seed + fold_id * 100_003,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_payload = run_domain_train_epoch(
            model=model,
            source_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            target_loader=target_loader,
            domain_classifier=domain_classifier,
        )
        train_loss = float(epoch_payload["train_loss"])
        val_score = -train_loss
        val_loss: float | None = None
        if val_rows:
            val_payload = predict_rows(
                model,
                val_rows,
                dataset_root=eval_dataset_root,
                preprocess=preprocess,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
            )
            if val_loader is not None:
                val_loss = evaluate_loss(model, val_loader, criterion, device)
            threshold = select_patient_threshold(
                rows=val_rows,
                y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
                strategy=str(args.threshold_strategy),
                default_threshold=float(args.threshold),
            )
            val_score = score_patient_threshold(
                rows=val_rows,
                y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
                threshold=threshold,
                metric="balanced_accuracy",
            )
        history.append(
            {
                "epoch": epoch,
                "val_loss": "" if val_loss is None else val_loss,
                "monitor_score": val_score,
                **epoch_payload,
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(args.patience):
            break
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    if bool(args.save_fold_models):
        torch.save(model.state_dict(), fold_dir / "model.pt")
    save_json(fold_dir / "preprocess.json", asdict(preprocess))
    write_csv(fold_dir / "history.csv", history)

    threshold = float(args.threshold)
    threshold_source = "fixed"
    if val_rows and str(args.threshold_strategy) != "fixed":
        val_payload = predict_rows(
            model,
            val_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        threshold = select_patient_threshold(
            rows=val_rows,
            y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
            strategy=str(args.threshold_strategy),
            default_threshold=float(args.threshold),
        )
        threshold_source = "validation"

    test_payload = predict_rows(
        model,
        test_rows,
        dataset_root=eval_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    lead_probs = lead_probabilities(test_rows, np.asarray(test_payload["y_prob"], dtype=np.float32))
    patient_probs = patient_condition_probabilities(
        test_rows,
        np.asarray(test_payload["y_prob"], dtype=np.float32),
    )
    pred_findings = {label: bool(patient_probs[label] >= threshold) for label in LABEL_NAMES}
    pred_label = findings_to_brugada(pred_findings)
    true_label = int(test_rows[0].reference_brugada)
    prob_derived_brugada = float(min(patient_probs.values()))
    gradcam_written = 0
    if write_gradcam:
        save_gradcam_panel(
            model=model,
            rows=test_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            device=device,
            output_path=fold_dir / "gradcam_panel.png",
            title=f"fold {fold_id} patient {test_rows[0].patient_id}",
        )
        gradcam_written = 1
    return {
        "patient_id": test_rows[0].patient_id,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob_derived_brugada": prob_derived_brugada,
        **{f"prob_{label.lower()}": patient_probs[label] for label in LABEL_NAMES},
        **{f"pred_{label.lower()}": int(pred_findings[label]) for label in LABEL_NAMES},
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "num_train_patients": len({row.patient_id for row in train_rows}),
        "num_train_images": len(train_rows),
        "domain_adaptation": str(args.domain_adaptation),
        "target_domain_patients": 0
        if target_rows is None
        else len({row.patient_id for row in target_rows}),
        "target_domain_images": 0 if target_rows is None else len(target_rows),
        "num_val_patients": len({row.patient_id for row in val_rows}),
        "num_val_images": len(val_rows),
        "lead_probs": json.dumps(lead_probs, sort_keys=True),
        "patient_condition_probs": json.dumps(patient_probs, sort_keys=True),
        "pred_findings": json.dumps(pred_findings, sort_keys=True),
        "gradcam_written": gradcam_written,
    }


def run_k_seed_shared_external_training(
    *,
    rows: Sequence[BrugadaImageRow],
    train_pool_rows: Sequence[BrugadaImageRow],
    folds: Sequence[dict[str, Any]],
    dataset_root: Path,
    train_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    run_dir: Path,
    k: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "fold_predictions.jsonl"
    resume = bool(getattr(args, "resume", True))
    existing = existing_keys(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    predictions: list[dict[str, object]] = load_existing_predictions(output_path)
    train_context_ids = select_qrs_training_patient_ids(
        train_pool_rows,
        k=k,
        seed=seed,
    )
    train_rows = rows_for_patient_ids(train_pool_rows, train_context_ids)
    model, preprocess, training_payload = train_qrs_model(
        train_rows=train_rows,
        train_dataset_root=train_dataset_root,
        target_dataset_root=target_dataset_root,
        target_rows=target_rows,
        run_dir=run_dir / "shared_training",
        seed=seed,
        args=args,
        device=device,
    )
    save_json(
        run_dir / "run_config.json",
        {
            "architecture": DEFAULT_MODEL_NAME,
            "external_train_mode": "shared",
            "train_dataset_root": train_dataset_root.as_posix(),
            "train_context_patient_ids": train_context_ids,
            "k": k,
            "seed": seed,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "threshold_strategy": str(args.threshold_strategy),
            "loss_pos_weight": str(args.loss_pos_weight),
            "resnet_weights": str(args.resnet_weights),
            "domain_adaptation": str(args.domain_adaptation),
            "domain_adaptation_weight": float(args.domain_adaptation_weight),
            "ssl_pretrain_epochs": int(args.ssl_pretrain_epochs),
            "target_dataset_root": ""
            if target_dataset_root is None
            else target_dataset_root.as_posix(),
            "target_domain_policy": "shared_transductive"
            if target_rows is not None
            else "none",
        },
    )

    gradcam_written = 0
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for(fold, k=k, seed=seed)
        val_rows = rows_for_patient_ids(rows, selection["validation_patient_ids"])
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        print(
            f"[INFO] CNN shared-train k={k} seed={seed} fold={fold_id} "
            f"test_patient={test_patient_id}"
        )
        result = predict_fold_with_trained_model(
            model=model,
            preprocess=preprocess,
            val_rows=val_rows,
            test_rows=test_rows,
            eval_dataset_root=dataset_root,
            fold_dir=run_dir / f"fold_{fold_id:04d}",
            fold_id=fold_id,
            seed=seed,
            args=args,
            device=device,
            write_gradcam=gradcam_written < int(args.gradcam_count),
        )
        gradcam_written += int(result.pop("gradcam_written", 0))
        result.update(
            {
                **training_payload,
                "key": key,
                "fold_id": fold_id,
                "k": k,
                "seed": seed,
                "test_patient_id": test_patient_id,
                "context_patient_ids": "|".join(train_context_ids),
                "validation_patient_ids": "|".join(selection["validation_patient_ids"]),
            }
        )
        append_jsonl(output_path, result)
        predictions.append(result)
        write_csv(run_dir / "fold_predictions.csv", predictions)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def train_qrs_model(
    *,
    train_rows: Sequence[BrugadaImageRow],
    train_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    run_dir: Path,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, PreprocessConfig, dict[str, object]]:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    preprocess = compute_preprocess_config(
        train_rows,
        dataset_root=train_dataset_root,
        image_size=int(args.image_size),
    )
    train_loader = make_loader(
        train_rows,
        dataset_root=train_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
    )
    model = build_model(resnet_weights=str(args.resnet_weights)).to(device)
    ssl_payload: dict[str, object] = {
        "ssl_pretrain_epochs": 0,
        "ssl_pretrain_seconds": 0.0,
    }
    if target_dataset_root is not None and target_rows and int(args.ssl_pretrain_epochs) > 0:
        encoder, ssl_payload = pretrain_encoder_simclr(
            target_rows=target_rows,
            target_dataset_root=target_dataset_root,
            preprocess=preprocess,
            run_dir=run_dir / "ssl_pretrain",
            seed=seed,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            resnet_weights=str(args.resnet_weights),
            epochs=int(args.ssl_pretrain_epochs),
            lr=float(args.ssl_pretrain_lr),
            weight_decay=float(args.weight_decay),
            projection_dim=int(args.ssl_projection_dim),
            temperature=float(args.ssl_temperature),
        )
        if encoder is not None:
            copy_encoder_weights(model, encoder)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=loss_pos_weight(train_rows).to(device)
        if str(args.loss_pos_weight) == "auto"
        else None
    )
    domain_classifier = (
        DomainClassifier().to(device) if str(args.domain_adaptation) == "dann" else None
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + ([] if domain_classifier is None else list(domain_classifier.parameters())),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    target_loader = make_target_loader(
        target_rows=target_rows,
        target_dataset_root=target_dataset_root,
        preprocess=preprocess,
        args=args,
        seed=seed,
        device=device,
    )
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_payload = run_domain_train_epoch(
            model=model,
            source_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            target_loader=target_loader,
            domain_classifier=domain_classifier,
        )
        history.append({"epoch": epoch, **epoch_payload})
    train_seconds = time.perf_counter() - started
    if bool(args.save_fold_models):
        torch.save(model.state_dict(), run_dir / "model.pt")
    save_json(run_dir / "preprocess.json", asdict(preprocess))
    write_csv(run_dir / "history.csv", history)
    return (
        model,
        preprocess,
        {
            "best_epoch": int(args.epochs),
            "train_seconds": train_seconds,
            "num_train_patients": len({row.patient_id for row in train_rows}),
            "num_train_images": len(train_rows),
            "domain_adaptation": str(args.domain_adaptation),
            "target_domain_patients": 0
            if target_rows is None
            else len({row.patient_id for row in target_rows}),
            "target_domain_images": 0 if target_rows is None else len(target_rows),
            **ssl_payload,
        },
    )


def predict_fold_with_trained_model(
    *,
    model: nn.Module,
    preprocess: PreprocessConfig,
    val_rows: Sequence[BrugadaImageRow],
    test_rows: Sequence[BrugadaImageRow],
    eval_dataset_root: Path,
    fold_dir: Path,
    fold_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    write_gradcam: bool,
) -> dict[str, object]:
    threshold = float(args.threshold)
    threshold_source = "fixed"
    if val_rows and str(args.threshold_strategy) != "fixed":
        val_payload = predict_rows(
            model,
            val_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        threshold = select_patient_threshold(
            rows=val_rows,
            y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
            strategy=str(args.threshold_strategy),
            default_threshold=float(args.threshold),
        )
        threshold_source = "validation"

    test_payload = predict_rows(
        model,
        test_rows,
        dataset_root=eval_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    lead_probs = lead_probabilities(test_rows, np.asarray(test_payload["y_prob"], dtype=np.float32))
    patient_probs = patient_condition_probabilities(
        test_rows,
        np.asarray(test_payload["y_prob"], dtype=np.float32),
    )
    pred_findings = {label: bool(patient_probs[label] >= threshold) for label in LABEL_NAMES}
    pred_label = findings_to_brugada(pred_findings)
    true_label = int(test_rows[0].reference_brugada)
    prob_derived_brugada = float(min(patient_probs.values()))
    gradcam_written = 0
    if write_gradcam:
        save_gradcam_panel(
            model=model,
            rows=test_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            device=device,
            output_path=fold_dir / "gradcam_panel.png",
            title=f"fold {fold_id} patient {test_rows[0].patient_id}",
        )
        gradcam_written = 1
    return {
        "patient_id": test_rows[0].patient_id,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob_derived_brugada": prob_derived_brugada,
        **{f"prob_{label.lower()}": patient_probs[label] for label in LABEL_NAMES},
        **{f"pred_{label.lower()}": int(pred_findings[label]) for label in LABEL_NAMES},
        "threshold": threshold,
        "threshold_source": threshold_source,
        "num_val_patients": len({row.patient_id for row in val_rows}),
        "num_val_images": len(val_rows),
        "lead_probs": json.dumps(lead_probs, sort_keys=True),
        "patient_condition_probs": json.dumps(patient_probs, sort_keys=True),
        "pred_findings": json.dumps(pred_findings, sort_keys=True),
        "gradcam_written": gradcam_written,
    }


def select_qrs_training_patient_ids(
    rows: Sequence[BrugadaImageRow],
    *,
    k: int,
    seed: int,
) -> list[str]:
    qrs_rows = [row for row in rows if row.has_qrs_labels]
    if not qrs_rows:
        raise ValueError("Training dataset must contain explicit QRS finding labels.")
    patients = patients_from_rows(qrs_rows)
    return select_context_patient_ids(
        patients,
        test_patient_id="__real_eval_patient__",
        k=k,
        seed=seed,
    )


def resolve_target_dataset_root(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    has_train_dataset: bool,
    needs_target_domain: bool,
) -> Path | None:
    explicit = str(args.target_dataset_root)
    if explicit not in {"", "."}:
        return Path(args.target_dataset_root).resolve()
    if needs_target_domain and has_train_dataset:
        return dataset_root
    if needs_target_domain:
        raise ValueError(
            "--target-dataset-root is required for domain adaptation when no "
            "external train dataset is used."
        )
    return None


def exclude_patient_rows(
    rows: Sequence[BrugadaImageRow] | None,
    patient_id: str,
) -> list[BrugadaImageRow] | None:
    if rows is None:
        return None
    return [row for row in rows if row.patient_id != patient_id]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def make_loader(
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ECGImageDataset(rows, dataset_root, preprocess),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def make_target_loader(
    *,
    target_rows: Sequence[BrugadaImageRow] | None,
    target_dataset_root: Path | None,
    preprocess: PreprocessConfig,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> DataLoader[torch.Tensor] | None:
    if str(args.domain_adaptation) == "none" or not target_rows or target_dataset_root is None:
        return None
    return make_unlabeled_loader(
        target_rows,
        dataset_root=target_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
    )


def build_resnet(resnet_weights: str) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if resnet_weights == "default" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_NAMES))
    return model


def build_model(*, resnet_weights: str) -> nn.Module:
    return build_resnet(resnet_weights)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(labels.shape[0])
        total += int(labels.shape[0])
    return total_loss / max(1, total)


def run_domain_train_epoch(
    *,
    model: nn.Module,
    source_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    target_loader: DataLoader[torch.Tensor] | None,
    domain_classifier: DomainClassifier | None,
) -> dict[str, float]:
    method = str(args.domain_adaptation)
    if method == "none" or target_loader is None:
        train_loss = run_train_epoch(model, source_loader, criterion, optimizer, device)
        return {
            "train_loss": train_loss,
            "supervised_loss": train_loss,
            "adaptation_loss": 0.0,
            "domain_weight": 0.0,
        }

    model.train()
    if domain_classifier is not None:
        domain_classifier.train()
    target_iter = cycle_batches(target_loader)
    total_loss = 0.0
    total_supervised = 0.0
    total_adaptation = 0.0
    total = 0
    weight = domain_adaptation_weight(
        epoch,
        int(args.epochs),
        float(args.domain_adaptation_weight),
    )
    kernel_scales = parse_float_list(str(args.mmd_kernel_scales))
    for source_images, labels in source_loader:
        source_images = source_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_images = next(target_iter).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        source_features = resnet_features(model, source_images)
        target_features = resnet_features(model, target_images)
        logits = resnet_logits_from_features(model, source_features)
        supervised_loss = criterion(logits, labels)

        if method == "coral":
            adaptation_loss = coral_loss(source_features, target_features)
            loss = supervised_loss + weight * adaptation_loss
        elif method == "mmd":
            adaptation_loss = mmd_loss(
                source_features,
                target_features,
                kernel_scales=kernel_scales,
            )
            loss = supervised_loss + weight * adaptation_loss
        elif method == "dann":
            if domain_classifier is None:
                raise RuntimeError("DANN requires a domain classifier.")
            features = torch.cat([source_features, target_features], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(source_features.shape[0], device=device),
                    torch.ones(target_features.shape[0], device=device),
                ]
            )
            domain_logits = domain_classifier(GradientReversal.apply(features, weight))
            adaptation_loss = F.binary_cross_entropy_with_logits(
                domain_logits,
                domain_labels,
            )
            loss = supervised_loss + adaptation_loss
        else:
            raise ValueError(f"Unsupported domain adaptation method: {method}")

        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_supervised += float(supervised_loss.item()) * batch_size
        total_adaptation += float(adaptation_loss.item()) * batch_size
        total += batch_size

    return {
        "train_loss": total_loss / max(1, total),
        "supervised_loss": total_supervised / max(1, total),
        "adaptation_loss": total_adaptation / max(1, total),
        "domain_weight": weight,
    }


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total += int(labels.shape[0])
    return total_loss / max(1, total)


def predict_rows(
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    _ = num_workers
    model.eval()
    y_true: list[list[int]] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(rows), max(1, batch_size)):
            batch_rows = list(rows[start : start + max(1, batch_size)])
            if not batch_rows:
                continue
            images = torch.stack(
                [
                    load_image_tensor(
                        dataset_root / row.image_path,
                        image_size=preprocess.image_size,
                        mean=preprocess.mean,
                        std=preprocess.std,
                    )
                    for row in batch_rows
                ]
            )
            logits = model(images.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(probs.astype(float).tolist())
            for row in batch_rows:
                if row.has_qrs_labels:
                    y_true.append([row.findings[label_name] for label_name in LABEL_NAMES])
    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_prob": np.asarray(y_prob, dtype=np.float64),
    }


def loss_pos_weight(rows: Sequence[BrugadaImageRow]) -> torch.Tensor:
    weights: list[float] = []
    for label_name in LABEL_NAMES:
        positives = sum(row.findings[label_name] for row in rows)
        negatives = len(rows) - positives
        weight = 1.0
        if positives > 0 and negatives > 0:
            weight = negatives / positives
        weights.append(float(weight))
    return torch.tensor(weights, dtype=torch.float32)


def select_patient_threshold(
    *,
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
    strategy: str,
    default_threshold: float,
) -> float:
    if strategy == "fixed" or y_prob.size == 0:
        return default_threshold
    true_binary = patient_reference_labels(rows)
    if np.unique(true_binary).size < 2:
        return default_threshold
    candidates = np.unique(
        np.concatenate(
            [
                y_prob.reshape(-1).astype(np.float64),
                np.linspace(0.05, 0.95, 19, dtype=np.float64),
                np.asarray([default_threshold], dtype=np.float64),
            ]
        )
    )
    best_threshold = default_threshold
    best_score = -math.inf
    metric_key = "f1" if strategy == "val_derived_f1" else "balanced_accuracy"
    for candidate in candidates:
        pred_binary = patient_predictions_at_threshold(rows, y_prob, float(candidate))
        score = brugada_metrics(true_binary, pred_binary).get(metric_key)
        if score is None:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_threshold = float(candidate)
    return best_threshold


def score_patient_threshold(
    *,
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
    threshold: float,
    metric: str,
) -> float:
    true_binary = patient_reference_labels(rows)
    pred_binary = patient_predictions_at_threshold(rows, y_prob, threshold)
    score = brugada_metrics(true_binary, pred_binary).get(metric)
    return float(score or 0.0)


def patient_reference_labels(rows: Sequence[BrugadaImageRow]) -> np.ndarray:
    by_patient: dict[str, int] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, int(row.reference_brugada))
    return np.asarray([by_patient[patient_id] for patient_id in sorted(by_patient)], dtype=np.int64)


def patient_predictions_at_threshold(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    predictions: dict[str, int] = {}
    for patient_id, probs in patient_probability_matrix(rows, y_prob).items():
        findings = {
            label_name: bool(probs[index] >= threshold)
            for index, label_name in enumerate(LABEL_NAMES)
        }
        predictions[patient_id] = findings_to_brugada(findings)
    return np.asarray(
        [predictions[patient_id] for patient_id in sorted(predictions)],
        dtype=np.int64,
    )


def patient_probability_matrix(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row, probs in zip(rows, y_prob, strict=True):
        grouped.setdefault(row.patient_id, []).append(np.asarray(probs, dtype=np.float64))
    return {
        patient_id: np.mean(np.vstack(values), axis=0)
        for patient_id, values in grouped.items()
    }


def patient_condition_probabilities(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
) -> dict[str, float]:
    matrix = patient_probability_matrix(rows, y_prob)
    if len(matrix) != 1:
        raise ValueError("Expected rows for exactly one patient.")
    probs = next(iter(matrix.values()))
    return {label_name: float(probs[index]) for index, label_name in enumerate(LABEL_NAMES)}


def lead_probabilities(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        row.lead: {label_name: float(probs[index]) for index, label_name in enumerate(LABEL_NAMES)}
        for row, probs in zip(rows, y_prob, strict=True)
    }


def summarize_predictions(
    predictions: Sequence[dict[str, object]],
    *,
    k: int,
    seed: int,
    run_dir: Path,
    report_dir: Path,
    model_name: str,
) -> dict[str, object]:
    y_true = np.asarray([int(row["true_label"]) for row in predictions], dtype=np.int64)
    y_pred = np.asarray([int(row["pred_label"]) for row in predictions], dtype=np.int64)
    y_prob = np.asarray(
        [float(row["prob_derived_brugada"]) for row in predictions], dtype=np.float64
    )
    metrics = brugada_metrics(y_true, y_pred)
    probability_metrics = probability_metrics_payload(
        y_true,
        y_prob,
        report_dir,
        prefix=f"cnn_k{k}_seed{seed}",
    )
    plot_confusion_matrix(run_dir / "confusion_matrix.png", metrics["counts"])
    save_json(run_dir / "metrics.json", {"metrics": metrics, **probability_metrics})
    return {
        "model_family": "cnn",
        "model": model_name,
        "k": k,
        "seed": seed,
        "n_patients": len(predictions),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "tp": metrics["counts"]["tp"],
        "tn": metrics["counts"]["tn"],
        "fp": metrics["counts"]["fp"],
        "fn": metrics["counts"]["fn"],
        "roc_auc": probability_metrics["roc_auc"],
        "average_precision": probability_metrics["average_precision"],
        "predictions": (run_dir / "fold_predictions.csv").as_posix(),
        "output_dir": run_dir.as_posix(),
    }


def probability_metrics_payload(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    report_dir: Path,
    *,
    prefix: str,
) -> dict[str, float | None]:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"roc_auc": None, "average_precision": None}
    roc_auc = float(roc_auc_score(y_true, y_prob))
    average_precision = float(average_precision_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    curves_dir = report_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_curve(
        curves_dir / f"{prefix}_roc.png",
        x=fpr,
        y=tpr,
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"CNN ROC {prefix}",
    )
    plot_curve(
        curves_dir / f"{prefix}_pr.png",
        x=recall,
        y=precision,
        xlabel="Recall",
        ylabel="Precision",
        title=f"CNN PR {prefix}",
    )
    return {"roc_auc": roc_auc, "average_precision": average_precision}


def write_summary_reports(rows: Sequence[dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "cnn_summary_by_seed.csv", rows)
    save_json(report_dir / "cnn_summary_by_seed.json", list(rows))
    by_k = aggregate_by_k(rows)
    write_csv(report_dir / "cnn_summary_by_k.csv", by_k)
    write_confusion_matrices_by_k(report_dir / "confusion_by_k", rows, prefix="cnn")
    plot_metric_by_k(
        report_dir / "balanced_accuracy_by_k.png",
        by_k,
        metric="balanced_accuracy_mean",
        ylabel="Balanced accuracy",
        title="CNN LOOCV balanced accuracy by k",
    )
    plot_metric_by_k(
        report_dir / "f1_by_k.png",
        by_k,
        metric="f1_mean",
        ylabel="F1",
        title="CNN LOOCV F1 by k",
    )
    print(f"[OK] Wrote CNN LOOCV report: {report_dir}")


def aggregate_by_k(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(row)
    out: list[dict[str, object]] = []
    for k, group in sorted(grouped.items()):
        payload: dict[str, object] = {"k": k, "n_runs": len(group)}
        for count_name in ("tp", "tn", "fp", "fn"):
            payload[count_name] = sum(int(row.get(count_name, 0) or 0) for row in group)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "roc_auc",
            "average_precision",
        ):
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) not in {None, "", "None"}
            ]
            payload[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            payload[f"{metric}_std"] = float(np.std(values, ddof=0)) if values else ""
        out.append(payload)
    return out


def write_confusion_matrices_by_k(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    *,
    prefix: str,
) -> None:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(row)
    for k, group in sorted(grouped.items()):
        counts = {
            name: sum(int(row.get(name, 0) or 0) for row in group)
            for name in ("tp", "tn", "fp", "fn")
        }
        plot_confusion_matrix(output_dir / f"{prefix}_k{k}_confusion_matrix.png", counts)


def save_gradcam_panel(
    *,
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    dataset_root: Path,
    preprocess: PreprocessConfig,
    device: torch.device,
    output_path: Path,
    title: str,
) -> None:
    target_module = gradcam_target_module(model)
    if target_module is None:
        return
    model.eval()
    gradcam = GradCAM(model=model, target_module=target_module)
    try:
        fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.2 * len(rows)))
        if len(rows) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_index, row in enumerate(rows):
            image_path = dataset_root / row.image_path
            input_tensor = load_image_tensor(
                image_path,
                image_size=preprocess.image_size,
                mean=preprocess.mean,
                std=preprocess.std,
            ).unsqueeze(0).to(device)
            display = load_raw_image(image_path, preprocess.image_size)
            with torch.enable_grad():
                heatmap = gradcam.compute(input_tensor)
            axes[row_index, 0].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 0].set_title(f"{row.lead} input")
            axes[row_index, 1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
            axes[row_index, 1].set_title("Grad-CAM")
            axes[row_index, 2].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 2].imshow(heatmap, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            axes[row_index, 2].set_title("Overlay")
            for col in range(3):
                axes[row_index, col].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    finally:
        gradcam.close()


def gradcam_target_module(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    return None


def plot_confusion_matrix(path: Path, counts: dict[str, int]) -> None:
    matrix = np.asarray([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curve(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_by_k(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    points = [(int(row["k"]), row.get(metric)) for row in rows if row.get(metric) != ""]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([point[0] for point in points], [float(point[1]) for point in points], marker="o")
    ax.set_xlabel("k patients")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["key"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_existing_predictions(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
