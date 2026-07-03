#!/usr/bin/env python3
"""Render the Brugada morphology schematic used in the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis" / "thesis" / "assets" / "results"

BLUE = "#005BBB"
RED = "#D71920"
BLACK = "#111111"
ANNOT = "#555555"
GRID_MINOR = "#F6DADA"
GRID_MAJOR = "#EDA6A6"


def gaussian(x: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def sigmoid(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


def normal_trace(x: np.ndarray) -> np.ndarray:
    y = -0.28 * np.ones_like(x)
    y += gaussian(x, 1.85, 0.28, 0.12)
    y += gaussian(x, 3.20, 0.08, -0.18)
    y += gaussian(x, 3.46, 0.11, 1.02)
    y += gaussian(x, 3.82, 0.18, -0.48)
    y += gaussian(x, 4.55, 0.14, 0.38)
    y += gaussian(x, 5.00, 0.16, -0.18)
    y += gaussian(x, 6.75, 0.54, 0.32)
    return y


def brugada_trace(x: np.ndarray) -> np.ndarray:
    y = 0.02 * np.ones_like(x)
    y += gaussian(x, 1.85, 0.30, 0.12)
    y += gaussian(x, 3.18, 0.10, -0.20)
    y += gaussian(x, 3.46, 0.12, 1.24)
    y += gaussian(x, 3.82, 0.20, -0.36)
    j_step = 0.58 * (sigmoid(x, 4.18, 0.06) - sigmoid(x, 4.72, 0.13))
    st_slope = 0.50 * (sigmoid(x, 4.68, 0.18) - sigmoid(x, 6.10, 0.42))
    y += j_step + st_slope
    y += gaussian(x, 6.70, 0.50, -0.34)
    y += gaussian(x, 7.95, 0.88, 0.08)
    return y


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.linewidth": 0.6,
            "savefig.dpi": 300,
        }
    )


def draw_ecg_grid(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 0.001, 1.0))
    ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 0.001, 0.2), minor=True)
    ax.set_yticks(np.arange(np.floor(ymin * 2) / 2, np.ceil(ymax * 2) / 2 + 0.001, 0.5))
    ax.set_yticks(np.arange(np.floor(ymin * 10) / 10, np.ceil(ymax * 10) / 10 + 0.001, 0.1), minor=True)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.35)
    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.75)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_annotation(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    align: str = "left",
) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha=align,
        va="center",
        color=BLACK,
        fontsize=8.2,
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOT,
            "linewidth": 0.65,
            "shrinkA": 2,
            "shrinkB": 3,
            "mutation_scale": 8,
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
    )


def main() -> None:
    setup_style()
    x = np.linspace(0, 9.0, 1600)
    y_normal = normal_trace(x)
    y_brugada = brugada_trace(x)

    fig, ax = plt.subplots(figsize=(6.2, 2.25))
    draw_ecg_grid(ax, 0, 9.0, -0.95, 1.35)

    ax.plot(x, y_normal, color=BLUE, linewidth=1.35, label="Normal/RBBB incompleto")
    ax.plot(x, y_brugada, color=RED, linewidth=1.6, label="Patrón tipo 1 simplificado")

    add_annotation(ax, "R' terminal", (3.46, y_normal[np.abs(x - 3.46).argmin()]), (2.82, 1.05), align="right")
    add_annotation(ax, "J elevado", (4.30, y_brugada[np.abs(x - 4.30).argmin()]), (4.72, 1.12))
    add_annotation(ax, "ST descendente", (5.20, y_brugada[np.abs(x - 5.20).argmin()]), (5.80, 0.74))
    add_annotation(ax, "T negativa", (6.70, y_brugada[np.abs(x - 6.70).argmin()]), (7.34, -0.42))

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        fancybox=False,
        edgecolor="#DDDDDD",
        borderpad=0.28,
        handlelength=2.6,
        columnspacing=1.3,
        prop={"size": 7.8},
    )

    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.04, top=0.86)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig_brugada_pattern.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
