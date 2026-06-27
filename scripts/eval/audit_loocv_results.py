#!/usr/bin/env python3
"""Audit QRS-derived Brugada LOOCV dataset, reports, and result completeness."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ecg_few.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    parse_int_list,
    read_jsonl,
    read_manifest,
    validate_fold_plan,
)

REQUIRED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
)
CNN_PROBABILITY_METRICS = ("roc_auc", "average_precision")
COUNTS = ("tp", "tn", "fp", "fn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LOOCV result completeness.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/brugada-huca/1.0.0"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument("--simulator-dataset-root", type=Path, default=Path("data/simulator_qrs"))
    parser.add_argument("--cnn-report-dir", type=Path, default=Path("reports/loocv/cnn"))
    parser.add_argument(
        "--cnn-simulator-report-dir",
        type=Path,
        default=Path("reports/loocv/cnn_simulator_qrs"),
    )
    parser.add_argument(
        "--cnn-comparison-dir",
        type=Path,
        default=Path("reports/loocv/cnn_comparison"),
    )
    parser.add_argument(
        "--domain-adaptation-report-dir",
        type=Path,
        default=Path("reports/loocv/cnn_domain_adaptation"),
    )
    parser.add_argument("--domain-adaptation-methods", default="ssl,coral,mmd,dann")
    parser.add_argument("--domain-k-values", default="16,32")
    parser.add_argument("--vlm-report-dir", type=Path, default=Path("reports/loocv/vlm"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("reports/loocv/comparison"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/loocv/audit"))
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--vlm-policy",
        choices=("todo", "required"),
        default="todo",
        help="Use 'todo' while VLM inference is intentionally not part of the final CNN package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)
    domain_k_values = parse_int_list(args.domain_k_values)
    report = audit(
        raw_root=args.raw_root,
        dataset_root=args.dataset_root,
        simulator_dataset_root=args.simulator_dataset_root,
        cnn_report_dir=args.cnn_report_dir,
        cnn_simulator_report_dir=args.cnn_simulator_report_dir,
        cnn_comparison_dir=args.cnn_comparison_dir,
        domain_adaptation_report_dir=args.domain_adaptation_report_dir,
        domain_adaptation_methods=parse_methods(args.domain_adaptation_methods),
        vlm_report_dir=args.vlm_report_dir,
        comparison_dir=args.comparison_dir,
        k_values=k_values,
        domain_k_values=domain_k_values,
        seeds=seeds,
        vlm_policy=str(args.vlm_policy),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "completeness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "analysis.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["status"], indent=2, sort_keys=True))


def audit(
    *,
    raw_root: Path,
    dataset_root: Path,
    simulator_dataset_root: Path,
    cnn_report_dir: Path,
    cnn_simulator_report_dir: Path,
    cnn_comparison_dir: Path,
    domain_adaptation_report_dir: Path,
    domain_adaptation_methods: list[str],
    vlm_report_dir: Path,
    comparison_dir: Path,
    k_values: list[int],
    domain_k_values: list[int],
    seeds: list[int],
    vlm_policy: str,
) -> dict[str, Any]:
    dataset = audit_dataset(
        raw_root=raw_root,
        dataset_root=dataset_root,
        k_values=k_values,
        seeds=seeds,
    )
    simulator_dataset = audit_simulator_dataset(
        dataset_root=simulator_dataset_root,
        k_values=k_values,
        seeds=seeds,
    )
    expected_clinical_patients = int(dataset.get("n_patients", 0) or 0)
    expected_simulator_patients = int(simulator_dataset.get("n_patients", 0) or 0)
    expected_runs = len(k_values) * len(seeds)
    cnn = audit_model_reports(
        report_dir=cnn_report_dir,
        model_family="cnn",
        expected_patients=expected_clinical_patients,
        expected_runs=expected_runs,
        k_values=k_values,
        seeds=seeds,
        require_probability_metrics=True,
        require_gradcam=True,
    )
    cnn_simulator = audit_model_reports(
        report_dir=cnn_simulator_report_dir,
        model_family="cnn",
        expected_patients=expected_simulator_patients,
        expected_runs=expected_runs,
        k_values=k_values,
        seeds=seeds,
        require_probability_metrics=True,
        require_gradcam=True,
    )
    cnn_comparison = audit_cnn_sim_real_comparison(cnn_comparison_dir, k_values=k_values)
    domain_adaptation = audit_domain_adaptation_reports(
        report_dir=domain_adaptation_report_dir,
        methods=domain_adaptation_methods,
        expected_patients=expected_clinical_patients,
        k_values=domain_k_values,
        seeds=seeds,
    )
    vlm_required = vlm_policy == "required"
    vlm = audit_model_reports(
        report_dir=vlm_report_dir,
        model_family="vlm",
        expected_patients=expected_clinical_patients,
        expected_runs=expected_runs,
        k_values=k_values,
        seeds=seeds,
        require_probability_metrics=False,
        require_gradcam=False,
    )
    if not vlm_required and not vlm["complete"]:
        vlm = {
            **vlm,
            "todo": True,
            "todo_items": list(vlm["errors"]),
            "errors": [],
            "warnings": [
                *vlm.get("warnings", []),
                "VLM inference is intentionally TODO for this CNN-only package.",
            ],
        }
    vlm_ready = audit_vlm_readiness(dataset_root=dataset_root)
    comparison = (
        audit_comparison(comparison_dir, cnn=cnn, vlm=vlm, k_values=k_values)
        if vlm_required
        else {
            "complete": True,
            "skipped": True,
            "errors": [],
            "warnings": ["CNN-vs-VLM comparison skipped because VLM is TODO."],
        }
    )
    all_cnn_complete = all(
        section["complete"]
        for section in (
            dataset,
            simulator_dataset,
            cnn,
            cnn_simulator,
            cnn_comparison,
            domain_adaptation,
        )
    )
    status = {
        "clinical_dataset_complete": dataset["complete"],
        "simulator_dataset_complete": simulator_dataset["complete"],
        "cnn_huca_results_complete": cnn["complete"],
        "cnn_simulator_results_complete": cnn_simulator["complete"],
        "cnn_sim_vs_real_comparison_complete": cnn_comparison["complete"],
        "cnn_domain_adaptation_complete": domain_adaptation["complete"],
        "all_cnn_experiments_complete": all_cnn_complete,
        "vlm_code_ready": vlm_ready["complete"],
        "vlm_results_status": "available" if vlm.get("summary_rows", 0) else "todo",
        "cnn_vs_vlm_comparison_complete": comparison["complete"],
        "final_cnn_package_ready": all_cnn_complete and vlm_ready["complete"],
    }
    return {
        "status": status,
        "expected": {
            "k_values": k_values,
            "domain_k_values": domain_k_values,
            "seeds": seeds,
            "runs_per_model": expected_runs,
            "clinical_patients_per_run": expected_clinical_patients,
            "simulator_patients_per_run": expected_simulator_patients,
            "clinical_fold_predictions_per_model": expected_runs * expected_clinical_patients,
            "simulator_fold_predictions_per_model": expected_runs * expected_simulator_patients,
        },
        "dataset": dataset,
        "simulator_dataset": simulator_dataset,
        "cnn": cnn,
        "cnn_simulator": cnn_simulator,
        "cnn_comparison": cnn_comparison,
        "domain_adaptation": domain_adaptation,
        "vlm": vlm,
        "vlm_readiness": vlm_ready,
        "comparison": comparison,
    }


def parse_methods(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def audit_dataset(
    *,
    raw_root: Path,
    dataset_root: Path,
    k_values: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    labels_dir = dataset_root / "labels"
    manifest_path = labels_dir / "all_labels.csv"
    excluded_path = labels_dir / "excluded_patients.csv"
    summary_path = labels_dir / "dataset_summary.json"
    folds_path = labels_dir / "loocv_folds.jsonl"
    rows = []
    folds = []
    if not raw_root.exists():
        errors.append(f"Missing raw Brugada-HUCA root: {raw_root}")
    for path in (manifest_path, excluded_path, summary_path, folds_path):
        if not path.exists():
            errors.append(f"Missing dataset artifact: {path}")
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest_path.exists():
        rows = read_manifest(manifest_path)
    if folds_path.exists():
        folds = read_jsonl(folds_path)
    if rows:
        lead_counts = Counter(row.patient_id for row in rows)
        bad_patients = sorted(patient for patient, count in lead_counts.items() if count != 3)
        if bad_patients:
            errors.append(f"Patients without exactly three rendered leads: {bad_patients[:10]}")
        labels = {row.reference_brugada for row in rows}
        if labels - {0, 1}:
            errors.append(f"Unsupported labels in manifest: {sorted(labels)}")
        borderline = [
            row.patient_id
            for row in rows
            if row.basal_pattern == 1 and row.clinical_brugada == 1
        ]
        if borderline:
            errors.append(f"Borderline positives still present: {sorted(set(borderline))[:10]}")
    if rows and folds:
        try:
            validate_fold_plan(folds, rows, k_values=k_values, seeds=seeds)
        except ValueError as exc:
            errors.append(str(exc))
    if summary:
        if summary.get("include_borderline_positive") is True:
            errors.append("Dataset summary says borderline positives were included.")
        if summary.get("right_precordial_leads") != ["V1", "V2", "V3"]:
            warnings.append("Dataset summary does not report V1/V2/V3 as the exact rendered leads.")
    return {
        "complete": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_patients": len({row.patient_id for row in rows}),
        "n_images": len(rows),
        "n_folds": len(folds),
        "clinical_label_counts": dict(Counter(str(row.reference_brugada) for row in rows)),
        "summary": summary,
    }


def audit_simulator_dataset(
    *,
    dataset_root: Path,
    k_values: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    labels_dir = dataset_root / "labels"
    manifest_path = labels_dir / "all_labels.csv"
    summary_path = labels_dir / "dataset_summary.json"
    folds_path = labels_dir / "loocv_folds.jsonl"
    for path in (manifest_path, summary_path, folds_path):
        if not path.exists():
            errors.append(f"Missing simulator artifact: {path}")
    rows = read_manifest(manifest_path) if manifest_path.exists() else []
    folds = read_jsonl(folds_path) if folds_path.exists() else []
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if rows:
        missing_qrs = [row for row in rows if not row.has_qrs_labels]
        if missing_qrs:
            errors.append("Simulator manifest contains rows without QRS finding labels.")
        lead_counts = Counter(row.patient_id for row in rows)
        bad_patients = sorted(patient for patient, count in lead_counts.items() if count != 3)
        if bad_patients:
            errors.append(
                f"Simulator patients without exactly three rendered leads: {bad_patients[:10]}"
            )
        labels = {row.reference_brugada for row in rows}
        if labels != {0, 1}:
            warnings.append(f"Simulator reference labels observed: {sorted(labels)}")
    if rows and folds:
        try:
            validate_fold_plan(folds, rows, k_values=k_values, seeds=seeds)
        except ValueError as exc:
            errors.append(str(exc))
    return {
        "complete": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_patients": len({row.patient_id for row in rows}),
        "n_images": len(rows),
        "n_folds": len(folds),
        "summary": summary,
    }


def audit_cnn_sim_real_comparison(
    comparison_dir: Path,
    *,
    k_values: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    table_path = comparison_dir / "cnn_simulated_vs_real_by_k.csv"
    if not table_path.exists():
        errors.append(f"Missing CNN sim-vs-real table: {table_path}")
    else:
        rows = read_csv(table_path)
        observed = {int(row["k"]) for row in rows if row.get("k")}
        if observed != set(k_values):
            errors.append(
                f"CNN sim-vs-real table has k={sorted(observed)}; expected {k_values}."
            )
    for plot_name in (
        "balanced_accuracy_simulated_vs_real_by_k.png",
        "f1_simulated_vs_real_by_k.png",
    ):
        if not (comparison_dir / plot_name).exists():
            errors.append(f"Missing CNN sim-vs-real plot: {comparison_dir / plot_name}")
    return {"complete": not errors, "errors": errors, "warnings": []}


def audit_domain_adaptation_reports(
    *,
    report_dir: Path,
    methods: list[str],
    expected_patients: int,
    k_values: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    expected_runs = len(k_values) * len(seeds)
    methods_payload: dict[str, Any] = {}
    for method in methods:
        methods_payload[method] = audit_model_reports(
            report_dir=report_dir / method,
            model_family="cnn",
            expected_patients=expected_patients,
            expected_runs=expected_runs,
            k_values=k_values,
            seeds=seeds,
            require_probability_metrics=True,
            require_gradcam=True,
        )
    comparison = audit_domain_adaptation_comparison(
        report_dir / "comparison",
        methods=methods,
        k_values=k_values,
    )
    complete = comparison["complete"] and all(
        payload["complete"] for payload in methods_payload.values()
    )
    errors = [
        f"{method}: {error}"
        for method, payload in methods_payload.items()
        for error in payload.get("errors", [])
    ]
    errors.extend(comparison.get("errors", []))
    warnings = [
        f"{method}: {warning}"
        for method, payload in methods_payload.items()
        for warning in payload.get("warnings", [])
    ]
    warnings.extend(comparison.get("warnings", []))
    return {
        "complete": complete,
        "errors": errors,
        "warnings": warnings,
        "methods": methods_payload,
        "comparison": comparison,
    }


def audit_domain_adaptation_comparison(
    comparison_dir: Path,
    *,
    methods: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    table_path = comparison_dir / "cnn_domain_adaptation_by_k.csv"
    if not table_path.exists():
        errors.append(f"Missing domain-adaptation comparison table: {table_path}")
    else:
        rows = read_csv(table_path)
        observed_methods = {row.get("method", "") for row in rows}
        expected_methods = {"baseline", *methods}
        missing_methods = expected_methods - observed_methods
        if missing_methods:
            errors.append(
                f"Domain-adaptation comparison misses methods: {sorted(missing_methods)}"
            )
        observed_k = {int(row["k"]) for row in rows if row.get("k")}
        if observed_k != set(k_values):
            errors.append(
                f"Domain-adaptation comparison has k={sorted(observed_k)}; "
                f"expected {k_values}."
            )
    for plot_name in (
        "balanced_accuracy_domain_adaptation_by_k.png",
        "f1_domain_adaptation_by_k.png",
    ):
        if not (comparison_dir / plot_name).exists():
            errors.append(f"Missing domain-adaptation plot: {comparison_dir / plot_name}")
    return {"complete": not errors, "errors": errors, "warnings": []}


def audit_model_reports(
    *,
    report_dir: Path,
    model_family: str,
    expected_patients: int,
    expected_runs: int,
    k_values: list[int],
    seeds: list[int],
    require_probability_metrics: bool,
    require_gradcam: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    summary_path = report_dir / f"{model_family}_summary_by_seed.csv"
    summary_by_k_path = report_dir / f"{model_family}_summary_by_k.csv"
    rows: list[dict[str, str]] = []
    if not summary_path.exists():
        errors.append(f"Missing {model_family} summary: {summary_path}")
    else:
        rows = read_csv(summary_path)
    if not summary_by_k_path.exists():
        errors.append(f"Missing {model_family} summary by k: {summary_by_k_path}")
    else:
        by_k_rows = read_csv(summary_by_k_path)
        if len(by_k_rows) != len(k_values):
            errors.append(f"{summary_by_k_path} should contain {len(k_values)} k rows.")
    if rows:
        expected_keys = {(str(k), str(seed)) for k in k_values for seed in seeds}
        observed_keys = {(row.get("k", ""), row.get("seed", "")) for row in rows}
        if observed_keys != expected_keys or len(rows) != expected_runs:
            errors.append(
                f"{model_family} summary has {len(rows)} runs; expected {expected_runs} "
                f"with k/seed grid {sorted(expected_keys)}."
            )
        required_columns = [*REQUIRED_METRICS, *COUNTS, "predictions"]
        if require_probability_metrics:
            required_columns.extend(CNN_PROBABILITY_METRICS)
        for column in required_columns:
            if column not in rows[0]:
                errors.append(f"{model_family} summary is missing column {column}.")
        for row in rows:
            run_label = f"k={row.get('k')} seed={row.get('seed')}"
            if int(float(row.get("n_patients", "0") or 0)) != expected_patients:
                errors.append(f"{model_family} {run_label} has incomplete patient count.")
            prediction_path = Path(row.get("predictions", ""))
            if not prediction_path.exists():
                errors.append(f"{model_family} {run_label} missing predictions: {prediction_path}")
                continue
            prediction_rows = read_csv(prediction_path)
            if len(prediction_rows) != expected_patients:
                errors.append(
                    f"{model_family} {run_label} has {len(prediction_rows)} predictions; "
                    f"expected {expected_patients}."
                )
            if required_prediction_columns_missing(prediction_rows):
                errors.append(f"{model_family} {run_label} prediction CSV lacks protocol columns.")
    for plot_name in ("balanced_accuracy_by_k.png",):
        if not (report_dir / plot_name).exists():
            errors.append(f"Missing {model_family} plot: {report_dir / plot_name}")
    for k in k_values:
        confusion_path = report_dir / "confusion_by_k" / f"{model_family}_k{k}_confusion_matrix.png"
        if not confusion_path.exists():
            errors.append(f"Missing {model_family} aggregated confusion matrix: {confusion_path}")
    if model_family == "cnn":
        if not (report_dir / "f1_by_k.png").exists():
            errors.append(f"Missing CNN F1 plot: {report_dir / 'f1_by_k.png'}")
        if rows:
            curves_dir = report_dir / "curves"
            missing_curves = [
                f"cnn_k{k}_seed{seed}_{suffix}.png"
                for k in k_values
                for seed in seeds
                for suffix in ("roc", "pr")
                if not (curves_dir / f"cnn_k{k}_seed{seed}_{suffix}.png").exists()
            ]
            if missing_curves:
                errors.append(f"Missing CNN ROC/PR curves: {missing_curves[:6]}")
    if require_gradcam:
        output_dirs = [Path(row["output_dir"]) for row in rows if row.get("output_dir")]
        gradcams = [
            path
            for output_dir in output_dirs
            if output_dir.exists()
            for path in output_dir.glob("fold_*/gradcam_panel.png")
        ]
        if not gradcams:
            errors.append("Missing CNN Grad-CAM panels under fold directories.")
    if model_family == "vlm" and not rows:
        warnings.append("VLM inference results are optional until Nahuel launches the heavy run.")
    return {
        "complete": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary_path": summary_path.as_posix(),
        "summary_rows": len(rows),
        "metrics_table": rows,
    }


def audit_vlm_readiness(dataset_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    for path in (
        dataset_root / "labels" / "all_labels.csv",
        dataset_root / "labels" / "loocv_folds.jsonl",
        Path("scripts/eval/run_vlm_loocv.py"),
        Path("scripts/eval/validate_vlm_loocv.py"),
        Path("prompts/system/qrs_huca.md"),
        Path("prompts/qrs/right_precordial_morphology.md"),
    ):
        if not path.exists():
            errors.append(f"Missing VLM readiness file: {path}")
    return {"complete": not errors, "errors": errors}


def audit_comparison(
    comparison_dir: Path,
    *,
    cnn: dict[str, Any],
    vlm: dict[str, Any],
    k_values: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    comparison_path = comparison_dir / "comparison_by_k.csv"
    plot_path = comparison_dir / "balanced_accuracy_by_k.png"
    if not comparison_path.exists():
        errors.append(f"Missing comparison table: {comparison_path}")
    else:
        rows = read_csv(comparison_path)
        if len(rows) != len(k_values):
            errors.append(f"Comparison table should contain {len(k_values)} rows.")
    if not plot_path.exists():
        errors.append(f"Missing comparison plot: {plot_path}")
    if cnn["complete"] and vlm["complete"] and not errors:
        errors.extend(compare_fold_alignment(cnn["metrics_table"], vlm["metrics_table"]))
    return {"complete": not errors, "errors": errors}


def compare_fold_alignment(
    cnn_rows: list[dict[str, str]],
    vlm_rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    cnn_by_key = {(row["k"], row["seed"]): row for row in cnn_rows}
    vlm_by_key = {(row["k"], row["seed"]): row for row in vlm_rows}
    if set(cnn_by_key) != set(vlm_by_key):
        return ["CNN/VLM summaries do not share exactly the same k/seed runs."]
    for key in sorted(cnn_by_key):
        cnn_fold_keys = fold_keys(read_csv(Path(cnn_by_key[key]["predictions"])))
        vlm_fold_keys = fold_keys(read_csv(Path(vlm_by_key[key]["predictions"])))
        if cnn_fold_keys != vlm_fold_keys:
            errors.append(f"CNN/VLM fold-context mismatch for k={key[0]}, seed={key[1]}.")
    return errors


def fold_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        (row["fold_id"], row["test_patient_id"], row["context_patient_ids"])
        for row in rows
    }


def required_prediction_columns_missing(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return True
    required = {"fold_id", "test_patient_id", "context_patient_ids", "true_label", "pred_label"}
    return bool(required.difference(rows[0]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_markdown(report: dict[str, Any]) -> str:
    status = report["status"]
    expected = report["expected"]
    dataset = report["dataset"]
    simulator_dataset = report["simulator_dataset"]
    lines = [
        "# LOOCV Results Audit",
        "",
        "## Status",
        "",
    ]
    for key, value in status.items():
        if isinstance(value, bool):
            rendered = "OK" if value else "MISSING"
        else:
            rendered = str(value).upper()
        lines.append(f"- {key}: {rendered}")
    lines.extend(
        [
            "",
            "## Datasets",
            "",
            f"- HUCA clinical patients: {dataset.get('n_patients')}",
            f"- HUCA clinical images: {dataset.get('n_images')}",
            f"- HUCA LOOCV folds: {dataset.get('n_folds')}",
            f"- Simulator patients: {simulator_dataset.get('n_patients')}",
            f"- Simulator images: {simulator_dataset.get('n_images')}",
            f"- Simulator LOOCV folds: {simulator_dataset.get('n_folds')}",
            f"- Expected runs per model: {expected.get('runs_per_model')}",
            (
                "- Expected HUCA fold predictions per model: "
                f"{expected.get('clinical_fold_predictions_per_model')}"
            ),
            (
                "- Expected simulator fold predictions per model: "
                f"{expected.get('simulator_fold_predictions_per_model')}"
            ),
            "",
            "## Interpretation",
            "",
        ]
    )
    if status["final_cnn_package_ready"]:
        lines.append(
            "All CNN datasets, reports, comparisons, and domain-adaptation summaries are "
            "complete for the configured grids. VLM is treated as a documented TODO unless "
            "`--vlm-policy required` is used."
        )
    else:
        lines.append(
            "The CNN package is not yet complete; inspect the sections below for missing "
            "artifacts or stale result grids."
        )
    lines.append("")
    for section_name in (
        "dataset",
        "simulator_dataset",
        "cnn",
        "cnn_simulator",
        "cnn_comparison",
        "domain_adaptation",
        "vlm",
        "vlm_readiness",
        "comparison",
    ):
        section = report[section_name]
        errors = section.get("errors", [])
        warnings = section.get("warnings", [])
        todo_items = section.get("todo_items", [])
        if errors or warnings:
            lines.extend([f"## {section_name.replace('_', ' ').title()}", ""])
            for error in errors:
                lines.append(f"- ERROR: {error}")
            for warning in warnings:
                lines.append(f"- WARNING: {warning}")
            lines.append("")
        if todo_items:
            lines.extend([f"## {section_name.replace('_', ' ').title()} TODO", ""])
            for item in todo_items:
                lines.append(f"- TODO: {item}")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
