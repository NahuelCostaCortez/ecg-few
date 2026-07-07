#!/usr/bin/env python3
# ruff: noqa: I001
"""Render VLM/ICL figures from Archive (10).zip.

The final thesis reports the zero-shot, normal ICL and balanced ICL
conditions, in that order.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "Archive (10).zip"
DEFAULT_RESULTS_DIR = Path.home() / "Downloads" / "vlm-results"
OUT = ROOT / "thesis" / "thesis" / "assets" / "results"

BLUE = "#005BBB"
RED = "#D71920"
GREEN = "#2E7D32"
ORANGE = "#D27D00"
BLACK = "#111111"
GRAY = "#666666"
GRID = "#B8B8B8"

TASKS = {
    "sim": {
        "label": "Simulador QRS/ST",
        "csv": "reports/loocv/vlm_simulator_qrs/vlm_summary_by_k.csv",
    },
    "real": {
        "label": "HUCA, V2",
        "csv": "reports/loocv/vlm_real_context/vlm_summary_by_k.csv",
    },
}


def main() -> None:
    args = parse_args()
    set_style()
    source = CsvSource(report_root=args.results_dir, zip_path=args.zip_path)
    rows = {key: selected_rows(source.read_csv(meta["csv"])) for key, meta in TASKS.items()}
    plot_icl_conditions(rows)
    plot_clinical_sensitivity_specificity(rows["real"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis VLM/ICL condition figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR if DEFAULT_RESULTS_DIR.exists() else None,
        help="Directory containing reports/loocv. Defaults to ~/Downloads/vlm-results if present.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=ZIP_PATH,
        help="Fallback archive containing reports/loocv.",
    )
    return parser.parse_args()


class CsvSource:
    def __init__(self, *, report_root: Path | None, zip_path: Path) -> None:
        self.report_root = report_root
        self.zip_path = zip_path

    def read_csv(self, member: str) -> list[dict[str, str]]:
        if self.report_root is not None:
            path = self.report_root / member
            if path.exists():
                return read_path_csv(path)
        if self.zip_path.exists():
            return read_zip_csv(self.zip_path, member)
        raise FileNotFoundError(
            "No VLM report source found. Provide --results-dir or --zip-path."
        )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 300,
        }
    )


def read_path_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_zip_csv(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(member).decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {"zero_shot": 0, "normal": 1, "balanced": 2}
    keep = [row for row in rows if row["condition"] in order]
    return sorted(keep, key=lambda row: (int(row["k"]), order[row["condition"]]))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else np.nan


def metric_series(
    rows: list[dict[str, str]], condition: str, metric: str
) -> tuple[list[int], list[float], list[float]]:
    ordered = [row for row in rows if row["condition"] == condition]
    ordered = sorted(ordered, key=lambda row: int(row["k"]))
    return (
        [int(row["k"]) for row in ordered],
        [f(row, f"{metric}_mean") for row in ordered],
        [f(row, f"{metric}_std") for row in ordered],
    )


def zero_value(rows: list[dict[str, str]], metric: str) -> tuple[float, float]:
    row = next(row for row in rows if row["condition"] == "zero_shot")
    return f(row, f"{metric}_mean"), f(row, f"{metric}_std")


def prepare_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1.5, 33.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0, 2, 4, 8, 16, 32])
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_condition_curves(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
) -> None:
    z_mean, z_std = zero_value(rows, metric)
    ax.errorbar(
        [0],
        [z_mean],
        yerr=[z_std],
        color=BLACK,
        marker="s",
        markersize=3.8,
        linewidth=0,
        capsize=2.5,
        elinewidth=0.75,
        label="zero-shot",
    )
    for condition, color, linestyle, marker, label in (
        ("normal", GREEN, "--", "o", "normal"),
        ("balanced", ORANGE, "-", "^", "balanceado"),
    ):
        k_values, means, stds = metric_series(rows, condition, metric)
        ax.errorbar(
            k_values,
            means,
            yerr=stds,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.2,
            linewidth=1.15,
            capsize=2.5,
            elinewidth=0.75,
            label=label,
        )


def plot_icl_conditions(rows: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.95, 4.35), sharex=True)

    panels = (
        (axes[0, 0], rows["sim"], "balanced_accuracy", "Simulador: BA", "Exactitud equilibrada"),
        (axes[0, 1], rows["sim"], "f1", "Simulador: F1", "F1"),
        (axes[1, 0], rows["real"], "balanced_accuracy", "HUCA V2: BA", "Exactitud equilibrada"),
        (axes[1, 1], rows["real"], "f1", "HUCA V2: F1", "F1"),
    )
    for ax, panel_rows, metric, title, ylabel in panels:
        plot_condition_curves(ax, panel_rows, metric)
        prepare_axis(ax, ylabel)
        ax.set_title(title, pad=3)

    for ax in axes[0, :]:
        ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        columnspacing=1.2,
        handlelength=1.7,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1), pad=0.35)

    save(fig, OUT / "vlm_icl_conditions_by_k.png")


def plot_clinical_sensitivity_specificity(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.65), sharex=True)

    for ax, metric, ylabel in (
        (axes[0], "sensitivity", "Sensibilidad"),
        (axes[1], "specificity", "Especificidad"),
    ):
        z_mean, z_std = zero_value(rows, metric)
        ax.errorbar(
            [0],
            [z_mean],
            yerr=[z_std],
            color=BLACK,
            marker="s",
            markersize=3.8,
            linewidth=0,
            capsize=2.5,
            elinewidth=0.75,
            label="zero-shot",
        )
        for condition, color, linestyle, marker, label in (
            ("normal", GREEN, "--", "o", "normal"),
            ("balanced", ORANGE, "-", "^", "balanceado"),
        ):
            k_values, means, stds = metric_series(rows, condition, metric)
            ax.errorbar(
                k_values,
                means,
                yerr=stds,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3.2,
                linewidth=1.15,
                capsize=2.5,
                elinewidth=0.75,
                label=label,
            )

        prepare_axis(ax, ylabel)
        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=1.0,
            borderpad=0.25,
            handlelength=1.4,
        )
    save(fig, OUT / "vlm_clinical_sensitivity_specificity.png")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fig.legends:
        fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
