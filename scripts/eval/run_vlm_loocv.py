#!/usr/bin/env python3
"""Run QRS-finding VLM ICL and derive Brugada with the shared LOOCV fold plan."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

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
from ecg_few.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    load_markdown_prompt,
    multilabel_answer_text,
    multilabel_json_schema,
)
from ecg_few.vlm.runtime import (
    LOCAL_RUNTIME,
    REMOTE_RUNTIME,
    LocalGPUGenerator,
    image_for_local,
    resolve_api_base,
    resolve_api_key,
    resolve_model_name,
)

ZERO_SHOT_CONDITION = "zero_shot"
NORMAL_CONDITION = "normal"
BALANCED_CONDITION = "balanced"
PERMUTED_CONDITION = "permuted"
NO_SUPPORT_IMAGES_CONDITION = "no_support_images"
ALL_CONDITIONS = (
    ZERO_SHOT_CONDITION,
    NORMAL_CONDITION,
    BALANCED_CONDITION,
    PERMUTED_CONDITION,
    NO_SUPPORT_IMAGES_CONDITION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM few-shot LOOCV for QRS findings and derived Brugada."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument(
        "--context-dataset-root",
        type=Path,
        default=Path(""),
        help="Optional synthetic QRS dataset used for ICL examples.",
    )
    parser.add_argument("--folds", type=Path, default=Path(""))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/vlm_loocv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/loocv/vlm"))
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model ids. Overrides --model when provided.",
    )
    parser.add_argument(
        "--conditions",
        default=NORMAL_CONDITION,
        help=(
            "Comma-separated VLM conditions: "
            "zero_shot,normal,balanced,permuted,no_support_images."
        ),
    )
    parser.add_argument(
        "--control-k-values",
        default="8,16,32",
        help="K values used by control conditions such as permuted and no_support_images.",
    )
    parser.add_argument(
        "--vlm-runtime",
        choices=(REMOTE_RUNTIME, LOCAL_RUNTIME),
        default=REMOTE_RUNTIME,
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--system-prompt-file", default="prompts/system/qrs_huca.md")
    parser.add_argument(
        "--prompt-file",
        default="prompts/qrs/right_precordial_morphology.md",
    )
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--local-device", default="cuda")
    parser.add_argument(
        "--local-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local-attn-implementation", default="sdpa")
    parser.add_argument("--local-offload-dir", type=Path, default=Path("outputs/vlm_loocv/offload"))
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--dry-run-predictions",
        choices=("none", "expected", "negative"),
        default="none",
        help="Write deterministic predictions without loading a model or calling an API.",
    )
    return parser.parse_args()


def main() -> None:
    run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    folds_path = resolve_folds_path(dataset_root, args.folds)
    rows = read_manifest(dataset_root / "labels" / "all_labels.csv")
    has_context_dataset = str(args.context_dataset_root) not in {"", "."}
    context_dataset_root = (
        Path(args.context_dataset_root).resolve()
        if has_context_dataset
        else dataset_root
    )
    context_pool_rows = (
        read_manifest(context_dataset_root / "labels" / "all_labels.csv")
        if has_context_dataset
        else rows
    )
    folds = read_jsonl(folds_path.resolve())
    k_values = parse_int_list(str(args.k_values))
    seeds = parse_int_list(str(args.seeds))
    fold_k_values = [k for k in k_values if k > 0]
    if fold_k_values:
        validate_fold_plan(folds, rows, k_values=fold_k_values, seeds=seeds)
    if int(args.limit_folds) > 0:
        folds = folds[: int(args.limit_folds)]

    runtime = str(args.vlm_runtime)
    models = resolve_models(args, runtime)
    conditions = parse_string_list(getattr(args, "conditions", NORMAL_CONDITION))
    invalid_conditions = sorted(set(conditions) - set(ALL_CONDITIONS))
    if invalid_conditions:
        raise ValueError(f"Unsupported VLM conditions: {invalid_conditions}")
    control_k_values = parse_int_list(str(getattr(args, "control_k_values", "8,16,32")))
    api_base = resolve_api_base(str(args.api_base), runtime)
    api_key = resolve_api_key(str(args.api_key), runtime)
    system_prompt = (
        load_markdown_prompt(args.system_prompt_file)
        if args.system_prompt_file
        else DEFAULT_SYSTEM_INSTRUCTIONS
    )
    prompt_template = load_markdown_prompt(args.prompt_file)
    output_root = Path(args.output_root).resolve()
    report_dir = Path(args.report_dir).resolve()

    summary_rows: list[dict[str, object]] = []
    for model in models:
        generator: LocalGPUGenerator | None = None
        if runtime == LOCAL_RUNTIME and args.dry_run_predictions == "none":
            generator = LocalGPUGenerator(
                model_id=model,
                device=str(args.local_device),
                dtype=str(args.local_dtype),
                attn_implementation=str(args.local_attn_implementation),
                offload_dir=Path(args.local_offload_dir) / model_slug(model),
                token=api_key,
            )
        try:
            for condition in conditions:
                condition_k_values = k_values_for_condition(
                    condition,
                    k_values=k_values,
                    control_k_values=control_k_values,
                )
                if not condition_k_values:
                    continue
                for k in condition_k_values:
                    for seed in seeds:
                        run_dir = (
                            output_root
                            / model_slug(model)
                            / condition
                            / f"k{k}_seed{seed}"
                        )
                        predictions = run_k_seed(
                            rows=rows,
                            context_pool_rows=context_pool_rows,
                            folds=folds,
                            dataset_root=dataset_root,
                            context_dataset_root=context_dataset_root,
                            run_dir=run_dir,
                            k=k,
                            seed=seed,
                            condition=condition,
                            runtime=runtime,
                            model=model,
                            api_base=api_base,
                            api_key=api_key,
                            system_prompt=system_prompt,
                            prompt_template=prompt_template,
                            args=args,
                            generator=generator,
                        )
                        summary_rows.append(
                            summarize_predictions(
                                predictions,
                                k=k,
                                seed=seed,
                                condition=condition,
                                model=model,
                                run_dir=run_dir,
                            )
                        )
        finally:
            if generator is not None:
                generator.close()
    write_summary_reports(summary_rows, report_dir)


def resolve_models(args: argparse.Namespace, runtime: str) -> list[str]:
    model_list = parse_string_list(str(getattr(args, "models", "") or ""))
    if model_list:
        return model_list
    return [resolve_model_name(str(getattr(args, "model", "")), runtime)]


def parse_string_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def model_slug(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def k_values_for_condition(
    condition: str,
    *,
    k_values: list[int],
    control_k_values: list[int],
) -> list[int]:
    if condition == ZERO_SHOT_CONDITION:
        return [0]
    positive = [k for k in k_values if k > 0]
    if condition in {PERMUTED_CONDITION, NO_SUPPORT_IMAGES_CONDITION}:
        controls = [k for k in control_k_values if k > 0]
        return [k for k in positive if k in set(controls)]
    return positive


def run_k_seed(
    *,
    rows: list[BrugadaImageRow],
    context_pool_rows: list[BrugadaImageRow],
    folds: list[dict[str, Any]],
    dataset_root: Path,
    context_dataset_root: Path,
    run_dir: Path,
    k: int,
    seed: int,
    condition: str,
    runtime: str,
    model: str,
    api_base: str | None,
    api_key: str,
    system_prompt: str,
    prompt_template: str,
    args: argparse.Namespace,
    generator: LocalGPUGenerator | None,
) -> list[dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "fold_predictions.jsonl"
    existing = existing_keys(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        output_path.unlink()
    predictions: list[dict[str, object]] = load_existing_predictions(output_path)
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{condition}:{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for_condition(
            fold=fold,
            rows=rows,
            context_pool_rows=context_pool_rows,
            test_patient_id=test_patient_id,
            k=k,
            seed=seed,
            fold_id=fold_id,
            condition=condition,
        )
        context_ids = selection["context_patient_ids"]
        if context_pool_rows is not rows and condition == NORMAL_CONDITION and k > 0:
            context_ids = select_qrs_context_patient_ids(
                context_pool_rows,
                k=k,
                seed=seed + fold_id * 100_003,
            )
        elif context_pool_rows is not rows and condition == BALANCED_CONDITION and k > 0:
            context_ids = select_balanced_context_patient_ids(
                context_pool_rows,
                test_patient_id="__real_eval_patient__",
                k=k,
                seed=seed + fold_id * 100_003,
            )
        context_rows = rows_for_patient_ids(context_pool_rows, context_ids)
        answer_rows = answer_rows_for_condition(
            context_rows,
            condition=condition,
            seed=seed + fold_id * 100_003,
        )
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        print(
            "[INFO] VLM "
            f"model={model} condition={condition} k={k} seed={seed} "
            f"fold={fold_id} test_patient={test_patient_id}"
        )
        started = time.perf_counter()
        lead_predictions: dict[str, dict[str, bool]] = {}
        lead_errors: dict[str, str] = {}
        raw_outputs: dict[str, str] = {}
        for test_row in test_rows:
            try:
                if args.dry_run_predictions != "none":
                    prediction = dry_run_prediction(test_row, mode=str(args.dry_run_predictions))
                    raw_text = json.dumps(prediction, sort_keys=True)
                else:
                    messages = build_messages(
                        dataset_root=dataset_root,
                        context_dataset_root=context_dataset_root,
                        system_prompt=system_prompt,
                        prompt_template=prompt_template,
                        context_rows=context_rows,
                        answer_rows=answer_rows,
                        test_row=test_row,
                        include_support_images=condition != NO_SUPPORT_IMAGES_CONDITION,
                        local=runtime == LOCAL_RUNTIME,
                    )
                    raw_text = call_model(
                        messages,
                        runtime=runtime,
                        model=model,
                        api_base=api_base,
                        api_key=api_key,
                        args=args,
                        generator=generator,
                    )
                    prediction = parse_prediction(raw_text)
                lead_predictions[test_row.lead] = {
                    label_name: bool(prediction[label_name]) for label_name in LABEL_NAMES
                }
                raw_outputs[test_row.lead] = raw_text
            except Exception as exc:  # noqa: BLE001
                lead_errors[test_row.lead] = str(exc)
                raw_outputs[test_row.lead] = ""
        pred_findings = aggregate_lead_findings(lead_predictions)
        pred_label = findings_to_brugada(pred_findings)
        record = {
            "key": key,
            "fold_id": fold_id,
            "k": k,
            "seed": seed,
            "condition": condition,
            "test_patient_id": test_patient_id,
            "patient_id": test_patient_id,
            "true_label": int(test_rows[0].reference_brugada),
            "pred_label": pred_label,
            **{f"pred_{label.lower()}": int(pred_findings[label]) for label in LABEL_NAMES},
            "context_patient_ids": "|".join(context_ids),
            "validation_patient_ids": "|".join(selection["validation_patient_ids"]),
            "valid_leads": len(lead_predictions),
            "invalid_leads": len(lead_errors),
            "json_invalid": int(bool(lead_errors)),
            "lead_predictions": json.dumps(lead_predictions, sort_keys=True),
            "raw_outputs": json.dumps(raw_outputs, sort_keys=True),
            "errors": json.dumps(lead_errors, sort_keys=True),
            "latency_seconds": time.perf_counter() - started,
            "model": model,
            "model_slug": model_slug(model),
            "vlm_runtime": runtime,
        }
        append_jsonl(output_path, record)
        predictions.append(record)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def selection_for_condition(
    *,
    fold: dict[str, Any],
    rows: list[BrugadaImageRow],
    context_pool_rows: list[BrugadaImageRow],
    test_patient_id: str,
    k: int,
    seed: int,
    fold_id: int,
    condition: str,
) -> dict[str, list[str]]:
    if k == 0 or condition == ZERO_SHOT_CONDITION:
        return {"context_patient_ids": [], "validation_patient_ids": []}
    if context_pool_rows is rows and condition == BALANCED_CONDITION:
        return {
            "context_patient_ids": select_balanced_context_patient_ids(
                rows,
                test_patient_id=test_patient_id,
                k=k,
                seed=seed + fold_id * 100_003,
            ),
            "validation_patient_ids": selection_for(fold, k=k, seed=seed)[
                "validation_patient_ids"
            ],
        }
    return selection_for(fold, k=k, seed=seed)


def answer_rows_for_condition(
    context_rows: list[BrugadaImageRow],
    *,
    condition: str,
    seed: int,
) -> list[BrugadaImageRow]:
    if condition != PERMUTED_CONDITION or len(context_rows) < 2:
        return context_rows
    shuffled = list(context_rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    unchanged = all(
        left.patient_id == right.patient_id and left.lead == right.lead
        for left, right in zip(context_rows, shuffled, strict=True)
    )
    if unchanged:
        shuffled = shuffled[1:] + shuffled[:1]
    return shuffled


def build_messages(
    *,
    dataset_root: Path,
    context_dataset_root: Path,
    system_prompt: str,
    prompt_template: str,
    context_rows: list[BrugadaImageRow],
    answer_rows: list[BrugadaImageRow],
    test_row: BrugadaImageRow,
    include_support_images: bool,
    local: bool,
) -> list[dict[str, Any]]:
    system_content: str | list[dict[str, str]]
    system_content = [{"type": "text", "text": system_prompt}] if local else system_prompt
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    for row, answer_row in zip(context_rows, answer_rows, strict=True):
        messages.append(
            support_user_message(
                prompt_for_row(prompt_template, row),
                context_dataset_root / row.image_path,
                include_image=include_support_images,
                local=local,
            )
        )
        messages.append(
            {
                "role": "assistant",
                "content": multilabel_answer_text(answer_row.expected_answer()),
            }
        )
    messages.append(
        user_message(
            prompt_for_row(prompt_template, test_row),
            dataset_root / test_row.image_path,
            local=local,
        )
    )
    return messages


def prompt_for_row(template: str, row: BrugadaImageRow) -> str:
    return template.replace("{lead}", row.lead)


def support_user_message(
    prompt: str,
    image_path: Path,
    *,
    include_image: bool,
    local: bool,
) -> dict[str, Any]:
    if include_image:
        return user_message(prompt, image_path, local=local)
    control_prompt = (
        f"{prompt}\n\n"
        "Control condition: the support image for this demonstration is intentionally "
        "omitted. Use only the label response that follows as context."
    )
    if local:
        return {"role": "user", "content": [{"type": "text", "text": control_prompt}]}
    return {"role": "user", "content": [{"type": "text", "text": control_prompt}]}


def user_message(prompt: str, image_path: Path, *, local: bool) -> dict[str, Any]:
    if local:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_for_local(image_path)},
            ],
        }
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url_for_image(image_path)}},
        ],
    }


def call_model(
    messages: list[dict[str, Any]],
    *,
    runtime: str,
    model: str,
    api_base: str | None,
    api_key: str,
    args: argparse.Namespace,
    generator: LocalGPUGenerator | None,
) -> str:
    if runtime == LOCAL_RUNTIME:
        if generator is None:
            raise RuntimeError("Local runtime was not initialized.")
        result = generator.generate_adaptive(
            [messages],
            initial_batch_size=int(args.local_batch_size),
            max_new_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
        )
        return result[0].texts[0]
    client = OpenAI(
        api_key=api_key or "EMPTY",
        base_url=str(api_base).rstrip("/") + "/",
        timeout=float(args.timeout),
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(args.max_output_tokens),
    }
    if float(args.temperature) != 0:
        payload["temperature"] = float(args.temperature)
    if args.response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "qrs_findings",
                "schema": multilabel_json_schema(),
                "strict": True,
            },
        }
    elif args.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return str(content).strip()


def dry_run_prediction(row: BrugadaImageRow, *, mode: str) -> dict[str, bool]:
    if mode == "expected":
        if not row.has_qrs_labels:
            raise ValueError("Dry-run expected mode requires QRS labels.")
        return row.expected_answer()
    if mode == "negative":
        return {label_name: False for label_name in LABEL_NAMES}
    raise ValueError(f"Unsupported dry-run mode: {mode}")


def parse_prediction(text: str) -> dict[str, bool]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and all(
            type(payload.get(label)) is bool for label in LABEL_NAMES
        ):
            return {label_name: bool(payload[label_name]) for label_name in LABEL_NAMES}
    raise ValueError("No valid QRS finding JSON prediction found.")


def aggregate_lead_findings(
    lead_predictions: dict[str, dict[str, bool]],
) -> dict[str, bool]:
    if not lead_predictions:
        return {label_name: False for label_name in LABEL_NAMES}
    return {
        label_name: (
            sum(1 for prediction in lead_predictions.values() if prediction[label_name])
            / len(lead_predictions)
            >= 0.5
        )
        for label_name in LABEL_NAMES
    }


def select_qrs_context_patient_ids(
    rows: list[BrugadaImageRow],
    *,
    k: int,
    seed: int,
) -> list[str]:
    qrs_rows = [row for row in rows if row.has_qrs_labels]
    if not qrs_rows:
        raise ValueError("Context dataset must contain explicit QRS finding labels.")
    patients = patients_from_rows(qrs_rows)
    return select_context_patient_ids(
        patients,
        test_patient_id="__real_eval_patient__",
        k=k,
        seed=seed,
    )


def select_balanced_context_patient_ids(
    rows: list[BrugadaImageRow],
    *,
    test_patient_id: str,
    k: int,
    seed: int,
) -> list[str]:
    qrs_rows = [row for row in rows if row.has_qrs_labels]
    if not qrs_rows:
        raise ValueError("Context dataset must contain explicit QRS finding labels.")
    patients = [
        patient
        for patient in patients_from_rows(qrs_rows)
        if patient.patient_id != str(test_patient_id)
    ]
    rng = random.Random(seed)
    by_label = {
        label: [patient for patient in patients if patient.reference_brugada == label]
        for label in (0, 1)
    }
    for candidates in by_label.values():
        candidates.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(candidates)
    n_positive = k // 2
    n_negative = k - n_positive
    selected = by_label[1][:n_positive] + by_label[0][:n_negative]
    selected_ids = {patient.patient_id for patient in selected}
    if len(selected) < k:
        remaining = [patient for patient in patients if patient.patient_id not in selected_ids]
        remaining.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    selected = selected[:k]
    return [patient.patient_id for patient in selected]


def summarize_predictions(
    predictions: list[dict[str, object]],
    *,
    k: int,
    seed: int,
    condition: str,
    model: str,
    run_dir: Path,
) -> dict[str, object]:
    y_true = [int(row["true_label"]) for row in predictions]
    y_pred = [int(row["pred_label"]) for row in predictions]
    metrics = brugada_metrics(y_true, y_pred)
    save_json(run_dir / "metrics.json", {"metrics": metrics})
    invalid_leads = sum(int(row.get("invalid_leads", 0) or 0) for row in predictions)
    total_leads = sum(
        int(row.get("valid_leads", 0) or 0) + int(row.get("invalid_leads", 0) or 0)
        for row in predictions
    )
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in predictions]
    return {
        "model_family": "vlm",
        "model": model,
        "model_slug": model_slug(model),
        "condition": condition,
        "k": k,
        "seed": seed,
        "n_patients": len(predictions),
        "n_leads": total_leads,
        "json_invalid_leads": invalid_leads,
        "json_invalid_rate": invalid_leads / total_leads if total_leads else 0.0,
        "latency_seconds_mean": float(np.mean(latencies)) if latencies else "",
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
        "roc_auc": "",
        "average_precision": "",
        "predictions": (run_dir / "fold_predictions.csv").as_posix(),
        "output_dir": run_dir.as_posix(),
    }


def write_summary_reports(rows: list[dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "vlm_summary_by_seed.csv", rows)
    save_json(report_dir / "vlm_summary_by_seed.json", rows)
    by_model_condition_k = aggregate_by_group(rows, group_keys=("model", "condition", "k"))
    write_csv(report_dir / "vlm_summary_by_k.csv", by_model_condition_k)
    write_csv(report_dir / "vlm_summary_by_model_condition_k.csv", by_model_condition_k)
    write_confusion_matrices_by_group(
        report_dir / "confusion_by_model_condition_k",
        rows,
        prefix="vlm",
    )
    plot_metric_by_group(
        report_dir / "balanced_accuracy_by_k.png",
        by_model_condition_k,
        metric="balanced_accuracy",
    )
    plot_metric_by_group(report_dir / "f1_by_k.png", by_model_condition_k, metric="f1")
    write_campaign_manifest(report_dir / "vlm_campaign_manifest.json", rows)
    print(f"[OK] Wrote VLM LOOCV report: {report_dir}")


def aggregate_by_group(
    rows: list[dict[str, object]],
    *,
    group_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: tuple(str(part) for part in item[0]),
    )
    for key, group in sorted_groups:
        payload: dict[str, object] = {
            group_key: value for group_key, value in zip(group_keys, key, strict=True)
        }
        payload["n_runs"] = len(group)
        for count_name in ("tp", "tn", "fp", "fn", "json_invalid_leads", "n_leads"):
            payload[count_name] = sum(int(row.get(count_name, 0) or 0) for row in group)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "json_invalid_rate",
            "latency_seconds_mean",
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


def write_confusion_matrices_by_group(
    output_dir: Path,
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model_slug"]), str(row["condition"]), int(row["k"]))
        grouped.setdefault(key, []).append(row)
    for (slug, condition, k), group in sorted(grouped.items()):
        counts = {
            name: sum(int(row.get(name, 0) or 0) for row in group)
            for name in ("tp", "tn", "fp", "fn")
        }
        filename = f"{prefix}_{slug}_{condition}_k{k}_confusion_matrix.png"
        plot_confusion_matrix(output_dir / filename, counts)


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


def plot_metric_by_group(path: Path, rows: list[dict[str, object]], *, metric: str) -> None:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        value = row.get(f"{metric}_mean")
        if value in {None, "", "None"}:
            continue
        key = (str(row.get("model", "")), str(row.get("condition", "")))
        grouped.setdefault(key, []).append((int(row["k"]), float(value)))
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, condition), points in sorted(grouped.items()):
        points = sorted(points)
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=f"{model} / {condition}",
        )
    ax.set_xlabel("k patients")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"VLM LOOCV {metric.replace('_', ' ')} by k")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_campaign_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    models = sorted({str(row.get("model", "")) for row in rows if row.get("model")})
    conditions = sorted({str(row.get("condition", "")) for row in rows if row.get("condition")})
    k_values = sorted({int(row["k"]) for row in rows}) if rows else []
    seeds = sorted({int(row["seed"]) for row in rows}) if rows else []
    payload = {
        "models": models,
        "conditions": conditions,
        "k_values": k_values,
        "seeds": seeds,
        "runs": len(rows),
        "artifacts": {
            "summary_by_seed": "vlm_summary_by_seed.csv",
            "summary_by_model_condition_k": "vlm_summary_by_model_condition_k.csv",
            "confusion_matrices": "confusion_by_model_condition_k/",
            "balanced_accuracy_plot": "balanced_accuracy_by_k.png",
            "f1_plot": "f1_by_k.png",
            "per_run_predictions": (
                "outputs/vlm_loocv/<model>/<condition>/"
                "k<k>_seed<seed>/fold_predictions.csv"
            ),
        },
    }
    save_json(path, payload)


def data_url_for_image(path: Path) -> str:
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
