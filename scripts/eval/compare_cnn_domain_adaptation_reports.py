#!/usr/bin/env python3
"""Compare CNN domain-adaptation summaries against the baseline CNN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "average_precision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN UDA reports by k.")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("reports/loocv/cnn/cnn_summary_by_k.csv"),
    )
    parser.add_argument(
        "--adaptation-root",
        type=Path,
        default=Path("reports/loocv/cnn_domain_adaptation"),
    )
    parser.add_argument(
        "--methods",
        default="ssl,coral,mmd,dann",
        help="Comma-separated adaptation report subdirectories to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/loocv/cnn_domain_adaptation/comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_rows = keyed_by_k(read_csv(args.baseline_summary))
    method_rows = {"baseline": baseline_rows}
    for method in parse_methods(args.methods):
        summary_path = args.adaptation_root / method / "cnn_summary_by_k.csv"
        if summary_path.exists():
            method_rows[method] = keyed_by_k(read_csv(summary_path))
    rows = build_comparison(method_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cnn_domain_adaptation_by_k.csv", rows)
    plot_metric(
        args.output_dir / "balanced_accuracy_domain_adaptation_by_k.png",
        rows,
        metric="balanced_accuracy",
        ylabel="Balanced accuracy",
    )
    plot_metric(
        args.output_dir / "f1_domain_adaptation_by_k.png",
        rows,
        metric="f1",
        ylabel="F1",
    )
    print(f"[OK] Wrote CNN domain-adaptation comparison under: {args.output_dir}")


def parse_methods(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed_by_k(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["k"]): row for row in rows}


def build_comparison(
    method_rows: dict[str, dict[int, dict[str, str]]],
) -> list[dict[str, object]]:
    adaptation_k_values = {
        k
        for method, rows_by_k in method_rows.items()
        if method != "baseline"
        for k in rows_by_k
    }
    k_values = sorted(
        adaptation_k_values
        or {
            k
            for rows_by_k in method_rows.values()
            for k in rows_by_k
        }
    )
    output: list[dict[str, object]] = []
    for k in k_values:
        for method, rows_by_k in method_rows.items():
            source_row = rows_by_k.get(k)
            if source_row is None:
                continue
            row: dict[str, object] = {
                "method": method,
                "k": k,
                "n_runs": source_row.get("n_runs", ""),
                "tp": source_row.get("tp", ""),
                "tn": source_row.get("tn", ""),
                "fp": source_row.get("fp", ""),
                "fn": source_row.get("fn", ""),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = source_row.get(f"{metric}_mean", "")
                row[f"{metric}_std"] = source_row.get(f"{metric}_std", "")
            output.append(row)
    return output


def plot_metric(path: Path, rows: list[dict[str, object]], *, metric: str, ylabel: str) -> None:
    if not rows:
        return
    methods = sorted({str(row["method"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method in methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["k"]),
        )
        k_values = [int(row["k"]) for row in method_rows]
        means = [float(row[f"{metric}_mean"]) for row in method_rows]
        stds = [float(row[f"{metric}_std"] or 0.0) for row in method_rows]
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=4, label=method)
    ax.set_xlabel("k synthetic training patients")
    ax.set_ylabel(ylabel)
    ax.set_title(f"CNN {ylabel.lower()} with unlabeled HUCA adaptation")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
