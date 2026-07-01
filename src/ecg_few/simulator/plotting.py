"""Plotting helpers for the ECG beat simulator."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from .simulator import generate_beat


def plot_beat(patient_beat, fs, ax=None, save_path=None, dpi=112):
    """Plot a single ECG beat with standard ECG grid styling."""
    target_pixels = 896
    size_in_inches = target_pixels / dpi

    if ax is None:
        fig, ax = plt.subplots(figsize=(size_in_inches, size_in_inches))
    else:
        fig = ax.get_figure()

    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    n_samples = patient_beat.shape[0]
    t = np.arange(n_samples) / fs * 1000

    ax.xaxis.set_minor_locator(MultipleLocator(40))
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

    ax.grid(True, which="major", color="pink", linewidth=0.8, alpha=0.5)
    ax.grid(True, which="minor", color="pink", linewidth=0.2, alpha=0.5)
    ax.minorticks_on()

    ax.plot(t, patient_beat, color="black", linewidth=2)
    ax.set_ylabel("Voltage (mV)")
    ax.set_xlabel("Time (ms)")

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    else:
        return (fig,)


def demo_plot_all(
    fs: int = 500,
    seed: int = 2026,
    save_path: str | None = "synthetic_ecg_examples.png",
) -> dict[str, dict[str, object]]:
    """Generate and plot one example beat per source family."""
    class_plan = [
        ("NORMAL", None),
        ("RBBB", None),
        ("ST_ELEVATION", None),
        ("T_WAVE_INVERSION", None),
        ("BRUGADA", "coved"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    outputs: dict[str, dict[str, object]] = {}

    for idx, (class_name, subtype) in enumerate(class_plan):
        beat, meta = generate_beat(class_name=class_name, seed=seed + idx, subtype=subtype, fs=fs)
        outputs[class_name] = {"waveform": beat, "metadata": meta}
        plot_beat(beat, fs=fs, ax=axes[idx])
        axes[idx].set_title(class_name if subtype is None else f"{class_name} ({subtype})")
        axes[idx].axhline(0.0, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)

    axes[-1].axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)

    return outputs


__all__ = ["plot_beat", "demo_plot_all"]