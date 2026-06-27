#!/usr/bin/env python3
"""Compare CNN and VLM LOOCV summaries and verify fold alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN and VLM LOOCV reports.")
    parser.add_argument(
        "--cnn-summary",
        type=Path,
        default=Path("reports/loocv/cnn/cnn_summary_by_seed.csv"),
    )
    parser.add_argument(
        "--vlm-summary",
        type=Path,
        default=Path("reports/loocv/vlm/vlm_summary_by_seed.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/loocv/comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cnn_rows = read_summary(args.cnn_summary)
    vlm_rows = read_summary(args.vlm_summary)
    verify_alignment(cnn_rows, vlm_rows)
    comparison = build_comparison(cnn_rows, vlm_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison_by_k.csv", comparison)
    plot_comparison(args.output_dir / "balanced_accuracy_by_k.png", comparison)
    print(f"[OK] Wrote LOOCV comparison under: {args.output_dir}")


def verify_alignment(cnn_rows: list[dict[str, object]], vlm_rows: list[dict[str, object]]) -> None:
    cnn_by_key = {(row["k"], row["seed"]): row for row in cnn_rows}
    vlm_by_key = {(row["k"], row["seed"]): row for row in vlm_rows}
    if set(cnn_by_key) != set(vlm_by_key):
        raise ValueError("CNN and VLM summaries do not contain the same k/seed runs.")
    for key in sorted(cnn_by_key):
        cnn_predictions = read_csv(Path(str(cnn_by_key[key]["predictions"])))
        vlm_predictions = read_csv(Path(str(vlm_by_key[key]["predictions"])))
        cnn_fold_keys = fold_keys(cnn_predictions)
        vlm_fold_keys = fold_keys(vlm_predictions)
        if cnn_fold_keys != vlm_fold_keys:
            raise ValueError(f"Fold/test/context mismatch for k={key[0]}, seed={key[1]}.")


def fold_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        (
            row["fold_id"],
            row["test_patient_id"],
            row["context_patient_ids"],
        )
        for row in rows
    }


def build_comparison(
    cnn_rows: list[dict[str, object]],
    vlm_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    cnn_by_k = aggregate_metric(cnn_rows)
    vlm_by_k = aggregate_metric(vlm_rows)
    output: list[dict[str, object]] = []
    for k in sorted(set(cnn_by_k) & set(vlm_by_k)):
        output.append(
            {
                "k": k,
                "cnn_balanced_accuracy_mean": format_metric(cnn_by_k[k]["mean"]),
                "cnn_balanced_accuracy_std": format_metric(cnn_by_k[k]["std"]),
                "vlm_balanced_accuracy_mean": format_metric(vlm_by_k[k]["mean"]),
                "vlm_balanced_accuracy_std": format_metric(vlm_by_k[k]["std"]),
            }
        )
    return output


def aggregate_metric(rows: list[dict[str, object]]) -> dict[int, dict[str, float | None]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        k = int(row["k"])
        grouped.setdefault(k, [])
        value = parse_optional_float(row.get("balanced_accuracy"))
        if value is not None:
            grouped[k].append(value)
    out: dict[int, dict[str, float | None]] = {}
    for k, values in grouped.items():
        if not values:
            out[k] = {"mean": None, "std": None}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        out[k] = {"mean": mean, "std": variance**0.5}
    return out


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    plot_rows = [
        row
        for row in rows
        if parse_optional_float(row["cnn_balanced_accuracy_mean"]) is not None
        and parse_optional_float(row["vlm_balanced_accuracy_mean"]) is not None
    ]
    if not plot_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    k_values = [int(row["k"]) for row in plot_rows]
    ax.plot(
        k_values,
        [float(row["cnn_balanced_accuracy_mean"]) for row in plot_rows],
        marker="o",
        label="CNN",
    )
    ax.plot(
        k_values,
        [float(row["vlm_balanced_accuracy_mean"]) for row in plot_rows],
        marker="o",
        label="VLM",
    )
    ax.set_xlabel("k patients")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("LOOCV few-shot comparison")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def read_summary(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of summary rows in {path}")
        return data
    return read_csv(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_metric(value: float | None) -> float | str:
    return "" if value is None else value


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
