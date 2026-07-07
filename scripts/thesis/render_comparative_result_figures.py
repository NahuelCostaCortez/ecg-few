#!/usr/bin/env python3
# ruff: noqa: I001
"""Render thesis-ready VLM and CNN/VLM comparison figures."""

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
OUT = ROOT / "thesis" / "thesis" / "assets" / "results"
DEFAULT_RESULTS_DIR = Path.home() / "Downloads" / "vlm-results"
DEFAULT_ZIP = ROOT / "Archive (10).zip"

MODEL = "google/gemma-4-E4B-it"
KEEP_CONDITIONS = ("zero_shot", "normal", "balanced")

BLUE = "#005BBB"
RED = "#D71920"
GREEN = "#2E7D32"
ORANGE = "#D27D00"
BLACK = "#111111"
GRAY = "#666666"
GRID = "#B8B8B8"
DOMAIN_GAP_YLIMS = {
    "balanced_accuracy": (0.42, 0.90),
    "f1": (0.00, 0.75),
}
DOMAIN_GAP_HUCA_COLORS = {
    BLUE: "#6FA8E8",
    GREEN: "#75B978",
    ORANGE: "#F0A53A",
}

DOMAIN_ADAPTATION_COLORS = {
    "baseline": BLACK,
    "ssl": GRAY,
    "coral": BLUE,
    "mmd": RED,
    "dann": "#7A4AB8",
}

CONDITION_STYLE = {
    "zero_shot": {
        "label": "zero-shot",
        "color": BLACK,
        "linestyle": "None",
        "marker": "s",
    },
    "normal": {"label": "ICL normal", "color": GREEN, "linestyle": "--", "marker": "o"},
    "balanced": {
        "label": "ICL balanceado",
        "color": ORANGE,
        "linestyle": "-",
        "marker": "^",
    },
}


def cnn_row(
    k: int,
    ba: float,
    ba_std: float,
    f1: float,
    f1_std: float,
    sensitivity: float,
    specificity: float,
    **counts: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "k": k,
        "balanced_accuracy_mean": ba,
        "balanced_accuracy_std": ba_std,
        "f1_mean": f1,
        "f1_std": f1_std,
        "sensitivity_mean": sensitivity,
        "specificity_mean": specificity,
    }
    row.update(counts)
    return row


CNN_SIM = [
    cnn_row(2, 0.625, 0.022, 0.401, 0.023, 0.650, 0.600),
    cnn_row(4, 0.631, 0.042, 0.407, 0.044, 0.617, 0.646),
    cnn_row(8, 0.685, 0.023, 0.468, 0.028, 0.683, 0.688),
    cnn_row(16, 0.773, 0.013, 0.569, 0.021, 0.800, 0.746),
    cnn_row(32, 0.848, 0.030, 0.673, 0.048, 0.883, 0.812),
]

CNN_REAL = [
    cnn_row(2, 0.509, 0.010, 0.484, 0.005, 0.693, 0.325),
    cnn_row(4, 0.509, 0.023, 0.493, 0.011, 0.730, 0.289),
    cnn_row(8, 0.490, 0.018, 0.469, 0.011, 0.672, 0.308),
    cnn_row(
        16,
        0.516,
        0.023,
        0.480,
        0.015,
        0.658,
        0.375,
        tn=226,
        fp=377,
        fn=119,
        tp=229,
    ),
    cnn_row(32, 0.499, 0.012, 0.466, 0.010, 0.644, 0.355),
]


def domain_row(
    method: str,
    k: int,
    accuracy: float,
    ba: float,
    ba_std: float,
    f1: float,
    precision: float,
    sensitivity: float,
    specificity: float,
) -> dict[str, object]:
    return {
        "method": method,
        "k": k,
        "accuracy_mean": accuracy,
        "balanced_accuracy_mean": ba,
        "balanced_accuracy_std": ba_std,
        "f1_mean": f1,
        "f1_std": "",
        "precision_mean": precision,
        "sensitivity_mean": sensitivity,
        "specificity_mean": specificity,
    }


CNN_DOMAIN_ADAPTATION = [
    domain_row("baseline", 16, 0.478, 0.516, 0.023, 0.480, 0.379, 0.658, 0.375),
    domain_row("ssl", 16, 0.429, 0.477, 0.027, 0.458, 0.351, 0.658, 0.297),
    domain_row("coral", 16, 0.493, 0.524, 0.008, 0.481, 0.384, 0.641, 0.408),
    domain_row("mmd", 16, 0.484, 0.516, 0.019, 0.474, 0.378, 0.635, 0.396),
    domain_row("dann", 16, 0.487, 0.515, 0.019, 0.470, 0.378, 0.621, 0.410),
    domain_row("baseline", 32, 0.461, 0.499, 0.012, 0.466, 0.366, 0.644, 0.355),
    domain_row("ssl", 32, 0.457, 0.503, 0.014, 0.476, 0.368, 0.672, 0.333),
    domain_row("coral", 32, 0.499, 0.528, 0.013, 0.480, 0.388, 0.632, 0.423),
    domain_row("mmd", 32, 0.522, 0.549, 0.029, 0.498, 0.404, 0.649, 0.448),
    domain_row("dann", 32, 0.473, 0.509, 0.028, 0.472, 0.373, 0.644, 0.375),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis comparison figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR if DEFAULT_RESULTS_DIR.exists() else None,
        help="Directory containing reports/loocv. Defaults to ~/Downloads/vlm-results if present.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP,
        help="Fallback archive containing VLM reports.",
    )
    parser.add_argument("--model", default=MODEL)
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


def main() -> None:
    args = parse_args()
    set_style()
    source = CsvSource(report_root=args.results_dir, zip_path=args.zip_path)
    sim_vlm = selected_vlm_rows(
        source.read_csv("reports/loocv/vlm_simulator_qrs/vlm_summary_by_model_condition_k.csv"),
        model=args.model,
    )
    real_vlm = selected_vlm_rows(
        source.read_csv("reports/loocv/vlm_real_context/vlm_summary_by_model_condition_k.csv"),
        model=args.model,
    )
    huca_sim_context_vlm = selected_vlm_rows(
        source.read_csv("reports/loocv/vlm/vlm_summary_by_model_condition_k.csv"),
        model=args.model,
    )

    plot_vlm_metric(
        OUT / "vlm_simulator_qrs_ba_by_k.png",
        sim_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        title="VLM/ICL en simulador QRS/ST",
    )
    plot_vlm_metric(
        OUT / "vlm_simulator_qrs_f1_by_k.png",
        sim_vlm,
        metric="f1",
        ylabel="F1",
        title="VLM/ICL en simulador QRS/ST",
    )
    plot_vlm_metric(
        OUT / "vlm_huca_v2_ba_by_k.png",
        real_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        title="VLM/ICL en V2 real",
    )
    plot_vlm_metric(
        OUT / "vlm_huca_v2_f1_by_k.png",
        real_vlm,
        metric="f1",
        ylabel="F1",
        title="VLM/ICL en V2 real",
    )
    plot_vlm_metric(
        OUT / "vlm_huca_synthetic_context_ba_by_k.png",
        huca_sim_context_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        title="Contexto sintético en datos reales",
    )
    plot_vlm_metric(
        OUT / "vlm_huca_synthetic_context_f1_by_k.png",
        huca_sim_context_vlm,
        metric="f1",
        ylabel="F1",
        title="Contexto sintético en datos reales",
    )

    plot_comparison_metric(
        OUT / "comparison_simulator_ba_by_k.png",
        CNN_SIM,
        sim_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        title="Comparación en simulador QRS/ST",
    )
    plot_comparison_metric(
        OUT / "comparison_simulator_f1_by_k.png",
        CNN_SIM,
        sim_vlm,
        metric="f1",
        ylabel="F1",
        title="Comparación en simulador QRS/ST",
    )
    plot_comparison_metric(
        OUT / "comparison_huca_ba_by_k.png",
        CNN_REAL,
        real_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        title="Comparación en datos reales",
    )
    plot_comparison_metric(
        OUT / "comparison_huca_f1_by_k.png",
        CNN_REAL,
        real_vlm,
        metric="f1",
        ylabel="F1",
        title="Comparación en datos reales",
    )
    plot_huca_sens_spec_best(OUT / "comparison_huca_sens_spec_best.png", real_vlm)
    plot_cnn_domain_gap_metric(
        OUT / "domain_gap_cnn_ba_by_k.png",
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
    )
    plot_cnn_domain_gap_metric(
        OUT / "domain_gap_cnn_f1_by_k.png",
        metric="f1",
        ylabel="F1",
    )
    plot_icl_domain_gap_metric(
        OUT / "domain_gap_icl_ba_by_k.png",
        sim_vlm,
        real_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
    )
    plot_icl_domain_gap_metric(
        OUT / "domain_gap_icl_f1_by_k.png",
        sim_vlm,
        real_vlm,
        metric="f1",
        ylabel="F1",
    )
    plot_icl_context_transfer_metric(
        OUT / "domain_gap_icl_context_ba_by_k.png",
        sim_vlm,
        real_vlm,
        huca_sim_context_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
    )
    plot_icl_context_transfer_metric(
        OUT / "domain_gap_icl_context_f1_by_k.png",
        sim_vlm,
        real_vlm,
        huca_sim_context_vlm,
        metric="f1",
        ylabel="F1",
    )
    plot_huca_context_sens_spec(
        OUT / "vlm_huca_context_sens_spec.png",
        real_vlm,
        huca_sim_context_vlm,
    )
    plot_domain_adaptation_metric(
        OUT / "cnn_domain_adaptation_balanced_accuracy_by_k.png",
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
    )
    plot_domain_adaptation_metric(
        OUT / "cnn_domain_adaptation_f1_by_k.png",
        metric="f1",
        ylabel="F1",
    )
    plot_transfer_controls_summary(
        OUT / "transfer_controls_synthetic_to_real_summary.png",
        huca_sim_context_vlm,
    )

    plot_confusion(
        OUT / "vlm_simulator_qrs_normal_k16_confusion_matrix.png",
        row_for(sim_vlm, "normal", 16),
        title="ICL normal, K=16",
    )
    plot_confusion(
        OUT / "vlm_simulator_qrs_balanced_k8_confusion_matrix.png",
        row_for(sim_vlm, "balanced", 8),
        title="ICL balanceado, K=8",
    )
    plot_confusion(
        OUT / "vlm_huca_v2_normal_k16_confusion_matrix.png",
        row_for(real_vlm, "normal", 16),
        title="ICL normal, K=16",
    )
    plot_confusion(
        OUT / "vlm_huca_v2_balanced_k16_confusion_matrix.png",
        row_for(real_vlm, "balanced", 16),
        title="ICL balanceado, K=16",
    )
    plot_confusion(
        OUT / "vlm_huca_synthetic_context_balanced_k32_confusion_matrix.png",
        row_for(huca_sim_context_vlm, "balanced", 32),
        title="Contexto sintético, K=32",
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


def selected_vlm_rows(rows: list[dict[str, str]], *, model: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("model") == model and row.get("condition") in KEEP_CONDITIONS
    ]
    if not selected:
        raise ValueError(f"No VLM rows found for model {model!r}")
    return sorted(
        selected,
        key=lambda row: (int(row["k"]), KEEP_CONDITIONS.index(row["condition"])),
    )


def f(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if value in ("", None):
        return np.nan
    return float(value)


def metric_key(metric: str, stat: str) -> str:
    return f"{metric}_{stat}"


def row_for(rows: list[dict[str, str]], condition: str, k: int) -> dict[str, str]:
    for row in rows:
        if row["condition"] == condition and int(row["k"]) == k:
            return row
    raise ValueError(f"Missing condition={condition} k={k}")


def rows_for_condition(rows: list[dict[str, str]], condition: str) -> list[dict[str, str]]:
    return sorted(
        [row for row in rows if row["condition"] == condition],
        key=lambda row: int(row["k"]),
    )


def prepare_axis(ax: plt.Axes, ylabel: str, *, include_zero: bool) -> None:
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0, 2, 4, 8, 16, 32] if include_zero else [2, 4, 8, 16, 32])
    ax.set_xlim((-1.5, 33.5) if include_zero else (0.5, 33.5))
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_vlm_metric(
    path: Path,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 2.45))
    for condition in KEEP_CONDITIONS:
        condition_rows = rows_for_condition(rows, condition)
        style = CONDITION_STYLE[condition]
        k_values = [int(row["k"]) for row in condition_rows]
        means = [f(row, metric_key(metric, "mean")) for row in condition_rows]
        stds = [f(row, metric_key(metric, "std")) for row in condition_rows]
        ax.errorbar(
            k_values,
            means,
            yerr=stds,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.2 if condition != "zero_shot" else 3.8,
            linewidth=1.15 if condition != "zero_shot" else 0,
            capsize=2.5,
            elinewidth=0.75,
            label=style["label"],
        )
    ax.set_title(title, pad=3)
    prepare_axis(ax, ylabel, include_zero=True)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25, handlelength=1.6)
    save(fig, path)


def plot_comparison_metric(
    path: Path,
    cnn_rows: list[dict[str, object]],
    vlm_rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.25, 2.55))
    k_values = [int(row["k"]) for row in cnn_rows]
    ax.errorbar(
        k_values,
        [f(row, metric_key(metric, "mean")) for row in cnn_rows],
        yerr=[f(row, metric_key(metric, "std")) for row in cnn_rows],
        color=BLUE,
        marker="o",
        markersize=3.2,
        linewidth=1.15,
        capsize=2.5,
        elinewidth=0.75,
        label="CNN",
    )
    for condition in ("normal", "balanced"):
        style = CONDITION_STYLE[condition]
        condition_rows = rows_for_condition(vlm_rows, condition)
        ax.errorbar(
            [int(row["k"]) for row in condition_rows],
            [f(row, metric_key(metric, "mean")) for row in condition_rows],
            yerr=[f(row, metric_key(metric, "std")) for row in condition_rows],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.2,
            linewidth=1.15,
            capsize=2.5,
            elinewidth=0.75,
            label=style["label"],
        )
    ax.set_title(title, pad=3)
    prepare_axis(ax, ylabel, include_zero=False)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25, handlelength=1.6)
    save(fig, path)


def plot_huca_sens_spec_best(path: Path, vlm_rows: list[dict[str, str]]) -> None:
    cnn = next(row for row in CNN_REAL if int(row["k"]) == 16)
    zero_shot = row_for(vlm_rows, "zero_shot", 0)
    normal = row_for(vlm_rows, "normal", 16)
    balanced = row_for(vlm_rows, "balanced", 16)
    rows = [
        ("CNN K=16", f(cnn, "sensitivity_mean"), f(cnn, "specificity_mean")),
        ("Zero-shot", f(zero_shot, "sensitivity_mean"), f(zero_shot, "specificity_mean")),
        ("ICL normal K=16", f(normal, "sensitivity_mean"), f(normal, "specificity_mean")),
        ("ICL bal. K=16", f(balanced, "sensitivity_mean"), f(balanced, "specificity_mean")),
    ]
    labels = [row[0] for row in rows]
    sensitivity = [row[1] for row in rows]
    specificity = [row[2] for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.6, 2.65))
    ax.bar(x - width / 2, sensitivity, width, label="Sensibilidad", color=RED)
    ax.bar(x + width / 2, specificity, width, label="Especificidad", color=BLUE)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_xticks(x, labels=labels, rotation=18, ha="right")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0, borderpad=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    save(fig, path)


def plot_cnn_domain_gap_metric(path: Path, *, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 2.45))
    plot_domain_pair(
        ax,
        CNN_SIM,
        CNN_REAL,
        metric=metric,
        label="CNN",
        color=BLUE,
        marker="o",
    )
    prepare_domain_gap_axis(ax, metric=metric, ylabel=ylabel)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25)
    save(fig, path)


def plot_icl_domain_gap_metric(
    path: Path,
    sim_vlm: list[dict[str, str]],
    real_vlm: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 2.45))
    for condition, marker in (("normal", "s"), ("balanced", "^")):
        style = CONDITION_STYLE[condition]
        plot_domain_pair(
            ax,
            rows_for_condition(sim_vlm, condition),
            rows_for_condition(real_vlm, condition),
            metric=metric,
            label=style["label"],
            color=style["color"],
            marker=marker,
        )
    prepare_domain_gap_axis(ax, metric=metric, ylabel=ylabel)
    ax.legend(
        loc="lower right",
        fontsize=6,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        handlelength=1.35,
    )
    save(fig, path)


def plot_icl_context_transfer_metric(
    path: Path,
    sim_vlm: list[dict[str, str]],
    real_vlm: list[dict[str, str]],
    huca_sim_context_vlm: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), sharey=True)
    for ax, condition, color, title in (
        (axes[0], "normal", GREEN, "ICL normal"),
        (axes[1], "balanced", ORANGE, "ICL balanceado"),
    ):
        condition_rows = (
            (rows_for_condition(sim_vlm, condition), "Simulador", "-", color, 1.0),
            (
                rows_for_condition(real_vlm, condition),
                "Real contexto real",
                "--",
                DOMAIN_GAP_HUCA_COLORS[color],
                1.0,
            ),
            (
                rows_for_condition(huca_sim_context_vlm, condition),
                "Real contexto sintético",
                ":",
                GRAY,
                1.0,
            ),
        )
        for rows, label, linestyle, line_color, alpha in condition_rows:
            ordered = sorted(rows, key=lambda row: int(row["k"]))
            ax.errorbar(
                [int(row["k"]) for row in ordered],
                [f(row, metric_key(metric, "mean")) for row in ordered],
                yerr=optional_yerr(ordered, metric),
                color=line_color,
                linestyle=linestyle,
                marker="o",
                markersize=3.0,
                linewidth=1.05,
                capsize=2.1,
                elinewidth=0.65,
                alpha=alpha,
                label=label,
            )
        ax.set_title(title, pad=3)
        prepare_domain_gap_axis(ax, metric=metric, ylabel=ylabel)
        ax.legend(
            loc="lower right",
            fontsize=5.8,
            frameon=True,
            framealpha=1.0,
            borderpad=0.25,
            handlelength=1.35,
        )
    axes[1].set_ylabel("")
    save(fig, path)


def plot_huca_context_sens_spec(
    path: Path,
    real_vlm: list[dict[str, str]],
    huca_sim_context_vlm: list[dict[str, str]],
) -> None:
    rows = [
        ("Real normal K=16", row_for(real_vlm, "normal", 16)),
        ("Real bal. K=16", row_for(real_vlm, "balanced", 16)),
        ("Sint. normal K=16", row_for(huca_sim_context_vlm, "normal", 16)),
        ("Sint. bal. K=32", row_for(huca_sim_context_vlm, "balanced", 32)),
    ]
    labels = [label for label, _ in rows]
    sensitivity = [f(row, "sensitivity_mean") for _, row in rows]
    specificity = [f(row, "specificity_mean") for _, row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.65, 2.7))
    ax.bar(x - width / 2, sensitivity, width, label="Sensibilidad", color=RED)
    ax.bar(x + width / 2, specificity, width, label="Especificidad", color=BLUE)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_xticks(x, labels=labels, rotation=18, ha="right")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0, borderpad=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    save(fig, path)


def plot_domain_adaptation_metric(path: Path, *, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    method_order = ("baseline", "ssl", "coral", "mmd", "dann")
    method_labels = {
        "baseline": "Base",
        "ssl": "SSL",
        "coral": "CORAL",
        "mmd": "MMD",
        "dann": "DANN",
    }
    for method in method_order:
        method_rows = sorted(
            [row for row in CNN_DOMAIN_ADAPTATION if row["method"] == method],
            key=lambda row: int(row["k"]),
        )
        ax.errorbar(
            [int(row["k"]) for row in method_rows],
            [f(row, metric_key(metric, "mean")) for row in method_rows],
            yerr=optional_yerr(method_rows, metric),
            color=DOMAIN_ADAPTATION_COLORS[method],
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            capsize=2.0,
            elinewidth=0.65,
            label=method_labels[method],
        )
    prepare_axis(ax, ylabel, include_zero=False)
    ax.set_xticks([16, 32])
    ax.set_xlim(13.5, 34.5)
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


def plot_transfer_controls_summary(
    path: Path,
    huca_sim_context_vlm: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.45), sharey="row")
    plot_domain_adaptation_on_axis(
        axes[0, 0],
        metric="balanced_accuracy",
        ylabel="Exactitud equilibrada",
        title="CNN con adaptación de dominio",
    )
    plot_vlm_on_axis(
        axes[0, 1],
        huca_sim_context_vlm,
        metric="balanced_accuracy",
        ylabel="",
        title="ICL con contexto sintético",
    )
    plot_domain_adaptation_on_axis(
        axes[1, 0],
        metric="f1",
        ylabel="F1",
        title="CNN con adaptación de dominio",
    )
    plot_vlm_on_axis(
        axes[1, 1],
        huca_sim_context_vlm,
        metric="f1",
        ylabel="",
        title="ICL con contexto sintético",
    )
    for ax in axes[0]:
        ax.set_ylim(0.45, 0.58)
    for ax in axes[1]:
        ax.set_ylim(0.0, 0.55)
    save(fig, path)


def plot_domain_adaptation_on_axis(
    ax: plt.Axes,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    method_order = ("baseline", "ssl", "coral", "mmd", "dann")
    method_labels = {
        "baseline": "Base",
        "ssl": "SSL",
        "coral": "CORAL",
        "mmd": "MMD",
        "dann": "DANN",
    }
    for method in method_order:
        method_rows = sorted(
            [row for row in CNN_DOMAIN_ADAPTATION if row["method"] == method],
            key=lambda row: int(row["k"]),
        )
        ax.errorbar(
            [int(row["k"]) for row in method_rows],
            [f(row, metric_key(metric, "mean")) for row in method_rows],
            yerr=optional_yerr(method_rows, metric),
            color=DOMAIN_ADAPTATION_COLORS[method],
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            capsize=2.0,
            elinewidth=0.65,
            label=method_labels[method],
        )
    ax.set_title(title, pad=3)
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([16, 32])
    ax.set_xlim(13.5, 34.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=5.8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        columnspacing=0.8,
        handlelength=1.25,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_vlm_on_axis(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for condition in KEEP_CONDITIONS:
        condition_rows = rows_for_condition(rows, condition)
        style = CONDITION_STYLE[condition]
        ax.errorbar(
            [int(row["k"]) for row in condition_rows],
            [f(row, metric_key(metric, "mean")) for row in condition_rows],
            yerr=optional_yerr(condition_rows, metric),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.0,
            linewidth=1.0 if condition != "zero_shot" else 0,
            capsize=2.0,
            elinewidth=0.65,
            label=style["label"],
        )
    ax.set_title(title, pad=3)
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 2, 4, 8, 16, 32])
    ax.set_xlim(-1.5, 33.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left" if metric == "f1" else "lower right",
        fontsize=5.8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        handlelength=1.25,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def optional_yerr(rows: list[dict[str, object]], metric: str) -> list[float] | None:
    values = [f(row, metric_key(metric, "std")) for row in rows]
    if all(np.isnan(value) for value in values):
        return None
    return values


def prepare_domain_gap_axis(ax: plt.Axes, *, metric: str, ylabel: str) -> None:
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*DOMAIN_GAP_YLIMS[metric])
    ax.set_xlim(0.5, 33.5)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_domain_pair(
    ax: plt.Axes,
    sim_rows: list[dict[str, object]],
    real_rows: list[dict[str, object]],
    *,
    metric: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    for rows, domain_label, linestyle, domain_color, alpha in (
        (sim_rows, "sintético", "-", color, 1.0),
        (real_rows, "datos reales", "--", DOMAIN_GAP_HUCA_COLORS[color], 1.0),
    ):
        ordered = sorted(rows, key=lambda row: int(row["k"]))
        ax.errorbar(
            [int(row["k"]) for row in ordered],
            [f(row, metric_key(metric, "mean")) for row in ordered],
            yerr=[f(row, metric_key(metric, "std")) for row in ordered],
            color=domain_color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.0,
            linewidth=1.05,
            capsize=2.2,
            elinewidth=0.65,
            alpha=alpha,
            label=f"{label} {domain_label}",
        )


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
