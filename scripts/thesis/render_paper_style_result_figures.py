#!/usr/bin/env python3
"""Render thesis result figures with a compact paper-like style."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "loocv"
OUT = ROOT / "thesis" / "thesis" / "assets" / "results"

BLUE = "#005BBB"
RED = "#D71920"
BLACK = "#111111"
GRAY = "#666666"
GRID = "#B8B8B8"


def main() -> None:
    set_style()

    sim_rows = read_csv(REPORTS / "cnn_simulator_qrs" / "cnn_summary_by_k.csv")
    real_rows = read_csv(REPORTS / "cnn" / "cnn_summary_by_k.csv")
    comparison_rows = read_csv(REPORTS / "cnn_comparison" / "cnn_simulated_vs_real_by_k.csv")
    domain_rows = read_csv(
        REPORTS / "cnn_domain_adaptation" / "comparison" / "cnn_domain_adaptation_by_k.csv"
    )

    plot_single_metric(
        OUT / "cnn_simulator_qrs_balanced_accuracy_by_k.png",
        sim_rows,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        label="CNN",
    )
    plot_single_metric(
        OUT / "cnn_huca_balanced_accuracy_by_k.png",
        real_rows,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        label="CNN",
    )
    plot_comparison(
        OUT / "cnn_balanced_accuracy_sim_vs_real.png",
        comparison_rows,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
    )
    plot_comparison(
        OUT / "cnn_f1_sim_vs_real.png",
        comparison_rows,
        metric="f1",
        ylabel="F1",
    )
    plot_domain_adaptation(
        OUT / "cnn_domain_adaptation_balanced_accuracy_by_k.png",
        domain_rows,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
    )
    plot_domain_adaptation(
        OUT / "cnn_domain_adaptation_f1_by_k.png",
        domain_rows,
        metric="f1",
        ylabel="F1",
    )

    plot_confusion(
        OUT / "cnn_simulator_qrs_k32_confusion_matrix.png",
        row_for_k(sim_rows, 32),
        title="Sintético, K=32",
    )
    plot_confusion(
        OUT / "cnn_huca_k16_confusion_matrix.png",
        row_for_k(real_rows, 16),
        title="HUCA, K=16",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_for_k(rows: list[dict[str, str]], k: int) -> dict[str, str]:
    for row in rows:
        if int(row["k"]) == k:
            return row
    raise ValueError(f"k={k} not found")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else np.nan


def ordered_by_k(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(row["k"]))


def prepare_axis(ax: plt.Axes, ylabel: str, ticks: list[int] | None = None) -> None:
    ticks = ticks or [2, 4, 8, 16, 32]
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(ticks)
    if len(ticks) == 2:
        ax.set_xlim(min(ticks) - 2, max(ticks) + 2)
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_single_metric(
    path: Path,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    label: str,
) -> None:
    rows = ordered_by_k(rows)
    k_values = [int(row["k"]) for row in rows]
    means = [as_float(row, f"{metric}_mean") for row in rows]
    stds = [as_float(row, f"{metric}_std") for row in rows]

    fig, ax = plt.subplots(figsize=(3.85, 2.35))
    ax.errorbar(
        k_values,
        means,
        yerr=stds,
        color=BLUE,
        marker="o",
        markersize=3.2,
        linewidth=1.15,
        capsize=2.5,
        elinewidth=0.75,
        label=label,
    )
    prepare_axis(ax, ylabel, sorted({int(row["k"]) for row in rows}))
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25, handlelength=1.4)
    save(fig, path)


def plot_comparison(
    path: Path,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    rows = ordered_by_k(rows)
    k_values = [int(row["k"]) for row in rows]

    fig, ax = plt.subplots(figsize=(4.25, 2.45))
    for prefix, label, color in (
        ("sim", "Sintético", BLUE),
        ("real", "HUCA", RED),
    ):
        means = [as_float(row, f"{prefix}_{metric}_mean") for row in rows]
        stds = [as_float(row, f"{prefix}_{metric}_std") for row in rows]
        ax.errorbar(
            k_values,
            means,
            yerr=stds,
            color=color,
            marker="o",
            markersize=3.0,
            linewidth=1.1,
            capsize=2.2,
            elinewidth=0.7,
            label=label,
        )
    prepare_axis(ax, ylabel)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25, handlelength=1.4)
    save(fig, path)


def plot_domain_adaptation(
    path: Path,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    order = ["baseline", "ssl", "coral", "mmd", "dann"]
    colors = {
        "baseline": BLACK,
        "ssl": GRAY,
        "coral": BLUE,
        "mmd": RED,
        "dann": "#7A4AB8",
    }

    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    for method in order:
        method_rows = ordered_by_k([row for row in rows if row["method"] == method])
        if not method_rows:
            continue
        k_values = [int(row["k"]) for row in method_rows]
        means = [as_float(row, f"{metric}_mean") for row in method_rows]
        stds = [as_float(row, f"{metric}_std") for row in method_rows]
        ax.errorbar(
            k_values,
            means,
            yerr=stds,
            color=colors[method],
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            capsize=2.0,
            elinewidth=0.65,
            label=method.upper() if method != "baseline" else "Base",
        )
    prepare_axis(ax, ylabel, sorted({int(row["k"]) for row in rows}))
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        columnspacing=0.8,
        handlelength=1.35,
    )
    save(fig, path)


def plot_confusion(path: Path, row: dict[str, str], *, title: str) -> None:
    matrix = np.array(
        [
            [int(row["tn"]), int(row["fp"])],
            [int(row["fn"]), int(row["tp"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(2.45, 2.35))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_title(title, pad=4)
    ax.set_xticks([0, 1], labels=["pred. 0", "pred. 1"])
    ax.set_yticks([0, 1], labels=["real 0", "real 1"])
    threshold = matrix.max() * 0.5
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > threshold else BLACK
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(width=0.55, length=2.5, labelsize=7)
    save(fig, path)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
