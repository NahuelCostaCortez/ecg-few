#!/usr/bin/env python3
"""Validate QRS-finding VLM LOOCV inputs without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecg_few.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    parse_int_list,
    patients_from_rows,
    read_jsonl,
    read_manifest,
    resolve_folds_path,
    select_context_patient_ids,
    selection_for,
    validate_fold_plan,
)
from ecg_few.prompts import load_markdown_prompt
from ecg_few.vlm.runtime import REMOTE_RUNTIME, resolve_api_base, resolve_model_name

ZERO_SHOT_CONDITION = "zero_shot"
NORMAL_CONDITION = "normal"
BALANCED_CONDITION = "balanced"
PERMUTED_CONDITION = "permuted"
NO_SUPPORT_IMAGES_CONDITION = "no_support_images"
ALL_CONDITIONS = {
    ZERO_SHOT_CONDITION,
    NORMAL_CONDITION,
    BALANCED_CONDITION,
    PERMUTED_CONDITION,
    NO_SUPPORT_IMAGES_CONDITION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate QRS-finding VLM LOOCV setup.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument("--context-dataset-root", type=Path, default=Path(""))
    parser.add_argument("--folds", type=Path, default=Path(""))
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--models", default="", help="Comma-separated model ids.")
    parser.add_argument(
        "--conditions",
        default=NORMAL_CONDITION,
        help="Comma-separated VLM conditions.",
    )
    parser.add_argument("--control-k-values", default="8,16,32")
    parser.add_argument("--vlm-runtime", default=REMOTE_RUNTIME)
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--system-prompt-file", default="prompts/system/qrs_huca.md")
    parser.add_argument(
        "--prompt-file",
        default="prompts/qrs/right_precordial_morphology.md",
    )
    parser.add_argument("--image-check", choices=("all", "none"), default="all")
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_setup(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if str(args.output):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not payload["ok"]:
        raise SystemExit(1)


def validate_setup(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_root = Path(args.dataset_root).resolve()
    has_context_dataset = str(args.context_dataset_root) not in {"", "."}
    context_dataset_root = (
        Path(args.context_dataset_root).resolve()
        if has_context_dataset
        else dataset_root
    )
    folds_path = resolve_folds_path(dataset_root, args.folds)
    k_values = parse_int_list(str(args.k_values))
    seeds = parse_int_list(str(args.seeds))
    conditions = parse_string_list(str(getattr(args, "conditions", NORMAL_CONDITION)))
    invalid_conditions = sorted(set(conditions) - ALL_CONDITIONS)
    if invalid_conditions:
        errors.append(f"Unsupported VLM conditions: {invalid_conditions}")
    control_k_values = parse_int_list(str(getattr(args, "control_k_values", "8,16,32")))
    models = resolve_models(args)
    rows = []
    context_rows = []
    folds = []
    try:
        rows = read_manifest(dataset_root / "labels" / "all_labels.csv")
        context_rows = (
            read_manifest(context_dataset_root / "labels" / "all_labels.csv")
            if has_context_dataset
            else rows
        )
        folds = read_jsonl(folds_path.resolve())
        positive_k_values = [k for k in k_values if k > 0]
        if positive_k_values:
            validate_fold_plan(folds, rows, k_values=positive_k_values, seeds=seeds)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    if int(args.limit_folds) > 0:
        folds = folds[: int(args.limit_folds)]
    if has_context_dataset and context_rows:
        if not any(row.has_qrs_labels for row in context_rows):
            errors.append("Context dataset has no QRS finding labels for ICL examples.")
    for prompt_path in (args.system_prompt_file, args.prompt_file):
        try:
            if not load_markdown_prompt(prompt_path).strip():
                errors.append(f"Prompt is empty: {prompt_path}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Invalid prompt {prompt_path}: {exc}")
    if args.vlm_runtime == REMOTE_RUNTIME:
        try:
            resolve_api_base(str(args.api_base), REMOTE_RUNTIME)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    if args.image_check == "all":
        for row in rows:
            image_path = dataset_root / row.image_path
            if not image_path.exists():
                errors.append(f"Missing image: {image_path}")
                break
        for row in context_rows:
            image_path = context_dataset_root / row.image_path
            if not image_path.exists():
                errors.append(f"Missing context image: {image_path}")
                break
    condition_k_plan = {
        condition: k_values_for_condition(
            condition,
            k_values=k_values,
            control_k_values=control_k_values,
        )
        for condition in conditions
    }
    planned_runs = (
        sum(len(values) for values in condition_k_plan.values()) * len(seeds) * len(models)
    )
    planned_patient_predictions = len(folds) * planned_runs
    planned_image_requests = planned_patient_predictions * 3
    context_examples = 0
    qrs_patients = patients_from_rows([row for row in context_rows if row.has_qrs_labels])
    for fold in folds:
        for condition, condition_k_values in condition_k_plan.items():
            for k in condition_k_values:
                for seed in seeds:
                    if k == 0 or condition == ZERO_SHOT_CONDITION:
                        context_ids = []
                    elif has_context_dataset:
                        context_ids = select_context_patient_ids(
                            qrs_patients,
                            test_patient_id="__real_eval_patient__",
                            k=k,
                            seed=seed + int(fold["fold_id"]) * 100_003,
                        )
                    else:
                        context_ids = selection_for(fold, k=k, seed=seed)[
                            "context_patient_ids"
                        ]
                    multiplier = 0 if condition == NO_SUPPORT_IMAGES_CONDITION else 3
                    context_examples += len(context_ids) * multiplier * len(models)
    return {
        "ok": not errors,
        "dataset_root": dataset_root.as_posix(),
        "context_dataset_root": context_dataset_root.as_posix(),
        "folds": len(folds),
        "patients": len({row.patient_id for row in rows}),
        "images": len(rows),
        "k_values": k_values,
        "seeds": seeds,
        "model": models[0]
        if models
        else resolve_model_name(str(args.model), str(args.vlm_runtime)),
        "models": models,
        "vlm_runtime": args.vlm_runtime,
        "conditions": conditions,
        "condition_k_plan": condition_k_plan,
        "planned_patient_predictions": planned_patient_predictions,
        "planned_image_requests": planned_image_requests,
        "planned_context_images": context_examples,
        "network_calls": 0,
        "model_loads": 0,
        "warnings": warnings,
        "errors": errors,
    }


def resolve_models(args: argparse.Namespace) -> list[str]:
    runtime = str(args.vlm_runtime)
    model_list = parse_string_list(str(getattr(args, "models", "") or ""))
    if model_list:
        return model_list
    return [resolve_model_name(str(getattr(args, "model", "")), runtime)]


def parse_string_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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
        controls = {k for k in control_k_values if k > 0}
        return [k for k in positive if k in controls]
    return positive


if __name__ == "__main__":
    main()
