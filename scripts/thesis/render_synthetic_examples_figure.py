#!/usr/bin/env python3
"""Render representative synthetic ECG examples for the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.colors import to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from ecg_few.simulator import generate_beat


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"

BLACK = "#111111"
GRID_MINOR = "#F6DADA"
GRID_MAJOR = "#EDA6A6"
BLUE = "#005BBB"
RED = "#D71920"
GREEN = "#4F8F3A"
GRAY = "#666666"
ATTENTION = "#D71920"
PREFERRED_SEEDS = {
    # Clear isolated ST elevation; the first accepted seed is visually too subtle.
    "ST_ELEVATION": 4735,
}


def main() -> None:
    set_style()
    examples = [
        ("NORMAL", "Normal", "sin hallazgos", "#F2F2F2"),
        ("RBBB", "RBBB", "R' terminal", "#DCE8F8"),
        ("ST_ELEVATION", "ST elevado", "J-ST elevado", "#FBE2D3"),
        ("T_WAVE_INVERSION", "T invertida", "T negativa", "#DDEDD5"),
        ("BRUGADA", "Brugada", "RBBB + ST + T", "#E9DDF1"),
    ]

    rows = []
    for family, title, subtitle, color in examples:
        waveform, metadata = accepted_waveform(family)
        rows.append((family, title, subtitle, color, waveform, metadata["feature_labels"]))

    fig = plt.figure(figsize=(6.45, 4.10))
    grid = fig.add_gridspec(
        len(rows),
        2,
        width_ratios=(0.19, 1.0),
        hspace=0.12,
        wspace=0.015,
    )

    strip_axes: list[plt.Axes] = []
    for idx, (family, title, subtitle, color, waveform, labels) in enumerate(rows):
        label_ax = fig.add_subplot(grid[idx, 0])
        share_ax = strip_axes[0] if strip_axes else None
        strip_ax = fig.add_subplot(grid[idx, 1], sharex=share_ax)
        strip_axes.append(strip_ax)

        draw_label(label_ax, title, subtitle, color)
        draw_strip(strip_ax, waveform, show_x=idx == len(rows) - 1)
        highlight_findings(strip_ax, labels)
        draw_attention_marks(strip_ax, family, waveform)

    strip_axes[-1].set_xlabel("Tiempo (ms)", labelpad=2, fontsize=8.2)
    fig.subplots_adjust(left=0.018, right=0.995, bottom=0.09, top=0.995)
    save(fig, OUT_DIR / "fig_synthetic_examples.png")
    save(fig, OUT_DIR / "fig_synthetic_examples.pdf")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.4,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.dpi": 300,
        }
    )


def accepted_waveform(family: str) -> tuple[np.ndarray, dict[str, object]]:
    subtype = "coved" if family == "BRUGADA" else None
    preferred_seed = PREFERRED_SEEDS.get(family)
    if preferred_seed is not None:
        waveform, metadata = generate_beat(
            class_name=family,
            seed=preferred_seed,
            subtype=subtype,
            fs=500,
            duration_ms=900,
        )
        labels = {key: int(value) for key, value in metadata["feature_labels"].items()}
        if family == "ST_ELEVATION" and labels.get("ST_ELEVATION") == 1:
            return waveform, metadata

    for seed in range(2026, 2600):
        waveform, metadata = generate_beat(
            class_name=family,
            seed=seed,
            subtype=subtype,
            fs=500,
            duration_ms=900,
        )
        labels = {key: int(value) for key, value in metadata["feature_labels"].items()}
        if family == "NORMAL" and not any(labels.values()):
            return waveform, metadata
        if family == "BRUGADA" and all(labels.values()):
            return waveform, metadata
        if family != "NORMAL" and family != "BRUGADA" and labels[family] == 1:
            return waveform, metadata
    raise RuntimeError(f"Could not generate accepted waveform for {family}")


def draw_strip(ax: plt.Axes, waveform: np.ndarray, *, show_x: bool) -> None:
    t_ms = np.arange(waveform.shape[0], dtype=float) / 500.0 * 1000.0
    ax.set_facecolor("#FFFAFA")
    ax.set_xlim(0, 900)
    ax.set_ylim(-1.08, 1.02)

    for x in np.arange(0, 901, 40):
        ax.axvline(x, color=GRID_MINOR, lw=0.35, zorder=0)
    for x in np.arange(0, 901, 200):
        ax.axvline(x, color=GRID_MAJOR, lw=0.68, zorder=0)
    for y in np.arange(-1.2, 1.21, 0.1):
        ax.axhline(y, color=GRID_MINOR, lw=0.35, zorder=0)
    for y in np.arange(-1.0, 1.1, 0.5):
        ax.axhline(y, color=GRID_MAJOR, lw=0.68, zorder=0)

    ax.axhline(0, color="#999999", lw=0.5, zorder=2)
    ax.plot(t_ms, waveform, color=BLACK, lw=1.25, zorder=4)

    ax.set_yticks([])
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.tick_params(axis="x", length=0, pad=1.5, colors=GRAY, labelbottom=show_x)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_label(ax: plt.Axes, title: str, subtitle: str, color: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.565),
            0.88,
            0.31,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color,
            edgecolor="#B8B8B8",
            linewidth=0.35,
        )
    )
    ax.text(
        0.08,
        0.72,
        title,
        ha="left",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=BLACK,
    )
    ax.text(0.08, 0.385, subtitle, ha="left", va="center", fontsize=6.9, color=GRAY)


def highlight_findings(ax: plt.Axes, labels: dict[str, int]) -> None:
    if int(labels.get("RBBB", 0)):
        ax.axvspan(300, 430, color=BLUE, alpha=0.08, zorder=1)
    if int(labels.get("ST_ELEVATION", 0)):
        ax.axvspan(392, 585, color=RED, alpha=0.08, zorder=1)
    if int(labels.get("T_WAVE_INVERSION", 0)):
        ax.axvspan(560, 830, color=GREEN, alpha=0.08, zorder=1)


def draw_attention_marks(ax: plt.Axes, family: str, waveform: np.ndarray) -> None:
    t_ms = np.arange(waveform.shape[0], dtype=float) / 500.0 * 1000.0
    windows = {
        "RBBB": [("max", 360, 470)],
        "ST_ELEVATION": [("at", 500, 500)],
        "T_WAVE_INVERSION": [("min", 590, 830)],
        "BRUGADA": [("max", 380, 470), ("at", 500, 500), ("min", 590, 830)],
    }
    points = [window_point(t_ms, waveform, mode, start, end) for mode, start, end in windows.get(family, [])]
    if not points:
        return

    for x, y in points:
        add_heat_spot(ax, x, y)


def add_heat_spot(ax: plt.Axes, x: float, y: float) -> None:
    size = 80
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    heat = np.exp(-2.8 * (xx**2 + yy**2))
    color = np.array(to_rgba(ATTENTION))
    rgba = np.zeros((size, size, 4), dtype=float)
    rgba[..., :3] = color[:3]
    rgba[..., 3] = 0.50 * heat
    ax.imshow(
        rgba,
        extent=(x - 30, x + 30, y - 0.40, y + 0.40),
        origin="lower",
        interpolation="bicubic",
        aspect="auto",
        zorder=3,
    )


def window_point(t_ms: np.ndarray, waveform: np.ndarray, mode: str, start: float, end: float) -> tuple[float, float]:
    mask = (t_ms >= start) & (t_ms <= end)
    if not np.any(mask):
        raise ValueError(f"Empty time window: {start}-{end} ms")
    values = waveform[mask]
    if mode == "at":
        idx = int(np.argmin(np.abs(t_ms[mask] - start)))
    else:
        idx = int(np.argmin(values) if mode == "min" else np.argmax(values))
    return float(t_ms[mask][idx]), float(values[idx])


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    if path.suffix.lower() == ".pdf":
        plt.close(fig)


if __name__ == "__main__":
    main()
