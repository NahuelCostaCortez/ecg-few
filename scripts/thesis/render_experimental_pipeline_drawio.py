#!/usr/bin/env python3
"""Render the experimental design pipeline as an editable Draw.io diagram."""

from __future__ import annotations

import base64
import csv
import html
import io
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps

from ecg_few.simulator import generate_beat


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"
EMBED_DIR = OUT_DIR / "drawio_embedded"
REPORTS = ROOT / "reports" / "loocv"
CLINICAL_EXAMPLE = ROOT / "data" / "brugada_huca" / "examples" / "clinical_brugada_example.png"
CNN_PIPELINE = OUT_DIR / "cnn_convolutional_pipeline.drawio.png"
VLM_PIPELINE = OUT_DIR / "vlm_icl_context_window.drawio.png"

WIDTH = 1180
HEIGHT = 520

INK = "#202124"
GRAY_STROKE = "#B8B8B8"
BLUE_FILL = "#DBE8FB"
BLUE_STROKE = "#7D94B5"
GREEN_FILL = "#D9EAD3"
GREEN_STROKE = "#93C47D"
PURPLE_FILL = "#EADFF1"
PURPLE_STROKE = "#9673A6"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    assets = {
        "synthetic": make_synthetic_thumb(),
        "clinical": make_clinical_thumb(),
        "cnn_pipeline": make_fitted_asset(CNN_PIPELINE, EMBED_DIR / "pipeline_cnn_drawio_panel.png", (390, 135)),
        "vlm_pipeline": make_fitted_asset(VLM_PIPELINE, EMBED_DIR / "pipeline_vlm_drawio_panel.png", (390, 135)),
        "curve": make_cnn_curve(),
        "confusion": make_confusion_matrix(),
    }

    write_drawio(assets)
    draw_preview(assets)
    print(OUT_DIR / "fig_pipeline_experimental.drawio")
    print(OUT_DIR / "fig_pipeline_experimental.drawio.png")


def fig_to_file_b64(fig: plt.Figure, path: Path, *, transparent: bool = False) -> tuple[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
    plt.close(fig)
    return image_file_b64(path)


def image_file_b64(path: Path) -> tuple[str, np.ndarray]:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), plt.imread(io.BytesIO(data))


def make_fitted_asset(source: Path, target: Path, size: tuple[int, int]) -> tuple[str, np.ndarray]:
    image = Image.open(source).convert("RGBA")
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (255, 255, 255, 0))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.alpha_composite(image, (x, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return image_file_b64(target)


def make_synthetic_thumb() -> tuple[str, np.ndarray]:
    waveform, _ = generate_beat(class_name="BRUGADA", seed=2029, subtype="coved", fs=500, duration_ms=900)
    t_ms = np.arange(waveform.shape[0], dtype=float) / 500.0 * 1000.0
    fig, ax = plt.subplots(figsize=(1.7, 1.55), dpi=220)
    draw_ecg_grid(ax, 0, 900, -1.05, 0.95)
    ax.plot(t_ms, waveform, color=INK, lw=1.35, zorder=4)
    ax.axis("off")
    return fig_to_file_b64(fig, EMBED_DIR / "pipeline_synthetic_ecg.png")


def make_clinical_thumb() -> tuple[str, np.ndarray]:
    image = Image.open(CLINICAL_EXAMPLE).convert("RGB")
    image = ImageOps.fit(image, (260, 240), method=Image.Resampling.LANCZOS, centering=(0.50, 0.50))
    path = EMBED_DIR / "pipeline_huca_ecg.png"
    image.save(path)
    return image_file_b64(path)


def make_cnn_curve() -> tuple[str, np.ndarray]:
    sim_rows = read_csv(REPORTS / "cnn_simulator_qrs" / "cnn_summary_by_k.csv")
    real_rows = read_csv(REPORTS / "cnn" / "cnn_summary_by_k.csv")
    fig, ax = plt.subplots(figsize=(2.25, 1.45), dpi=220)
    for rows, label, color in (
        (sim_rows, "Sint.", "#005BBB"),
        (real_rows, "HUCA", "#D71920"),
    ):
        rows = sorted(rows, key=lambda row: int(row["k"]))
        k = [int(row["k"]) for row in rows]
        y = [float(row["balanced_accuracy_mean"]) for row in rows]
        ax.plot(k, y, marker="o", lw=1.25, ms=2.8, color=color, label=label)
    ax.set_ylim(0.35, 0.95)
    ax.set_xticks([2, 8, 16, 32])
    ax.set_yticks([0.4, 0.6, 0.8])
    ax.grid(True, color="#C9C9C9", lw=0.45)
    ax.tick_params(labelsize=6, length=2, width=0.5, pad=1)
    ax.set_xlabel("K", fontsize=6, labelpad=1)
    ax.set_ylabel("BA", fontsize=6, labelpad=1)
    ax.legend(loc="lower right", frameon=True, fontsize=5.4, borderpad=0.2, handlelength=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    return fig_to_file_b64(fig, EMBED_DIR / "pipeline_cnn_curve.png")


def make_confusion_matrix() -> tuple[str, np.ndarray]:
    row = sorted(read_csv(REPORTS / "cnn_simulator_qrs" / "cnn_summary_by_k.csv"), key=lambda item: int(item["k"]))[-1]
    matrix = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
    fig, ax = plt.subplots(figsize=(1.15, 1.15), dpi=220)
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.tick_params(length=0, labelsize=6)
    threshold = matrix.max() * 0.55
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > threshold else INK
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6.3, color=color)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    return fig_to_file_b64(fig, EMBED_DIR / "pipeline_confusion.png")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_ecg_grid(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    ax.set_facecolor("#FFFAFA")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for x in np.arange(xmin, xmax + 1, 40):
        ax.axvline(x, color="#F6DADA", lw=0.35, zorder=0)
    for x in np.arange(xmin, xmax + 1, 200):
        ax.axvline(x, color="#EDA6A6", lw=0.7, zorder=0)
    for y in np.arange(ymin - 0.2, ymax + 0.2, 0.1):
        ax.axhline(y, color="#F6DADA", lw=0.35, zorder=0)
    for y in np.arange(-1.0, 1.1, 0.5):
        ax.axhline(y, color="#EDA6A6", lw=0.7, zorder=0)


def rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, fill: str, stroke: str, lw: float = 1.1, radius: float = 8) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
        )
    )


def label(ax: plt.Axes, x: float, y: float, text: str, **kwargs) -> None:
    opts = {
        "ha": "center",
        "va": "center",
        "fontfamily": "Arial",
        "fontsize": 10,
        "color": INK,
    }
    opts.update(kwargs)
    ax.text(x, y, text, **opts)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = INK, lw: float = 2.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, header: str, fill: str, stroke: str, *, header_fill: str = BLUE_FILL, header_stroke: str = BLUE_STROKE) -> None:
    rounded(ax, x + 5, y + 7, w, h, "#ECEFF3", "#ECEFF3", lw=0.0, radius=11)
    rounded(ax, x, y, w, h, fill, stroke, lw=1.1, radius=11)
    header_w = min(w - 42, max(120, len(header) * 8.4 + 34))
    rounded(ax, x + (w - header_w) / 2, y - 18, header_w, 35, header_fill, header_stroke, lw=1.0, radius=4)
    label(ax, x + w / 2, y - 0.5, header, fontsize=10.8, fontweight="bold")


def image(ax: plt.Axes, img: np.ndarray, x: float, y: float, w: float, h: float) -> None:
    ax.imshow(img, extent=(x, x + w, y + h, y), aspect="auto", zorder=4)


def mini_pill(ax: plt.Axes, x: float, y: float, w: float, text: str, fill: str, stroke: str) -> None:
    rounded(ax, x, y, w, 24, fill, stroke, lw=0.8, radius=5)
    label(ax, x + w / 2, y + 12, text, fontsize=8.2, fontweight="bold")


def draw_preview(assets: dict[str, tuple[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    card(ax, 48, 150, 260, 245, "A: Entrada común", "#FFFFFF", "#8A8A8A")
    mini_pill(ax, 76, 184, 96, "Sintético", BLUE_FILL, BLUE_STROKE)
    mini_pill(ax, 188, 184, 84, "HUCA", GREEN_FILL, GREEN_STROKE)
    image(ax, assets["synthetic"][1], 72, 226, 103, 100)
    image(ax, assets["clinical"][1], 186, 226, 96, 100)

    card(ax, 402, 58, 430, 190, "B: CNN", "#FFFFFF", "#6F6F6F", header_fill=BLUE_FILL, header_stroke=BLUE_STROKE)
    image(ax, assets["cnn_pipeline"][1], 422, 97, 390, 135)

    card(ax, 402, 312, 430, 190, "C: VLM/ICL", "#FFFFFF", "#6F6F6F", header_fill=PURPLE_FILL, header_stroke=PURPLE_STROKE)
    image(ax, assets["vlm_pipeline"][1], 422, 352, 390, 135)

    card(ax, 875, 130, 305, 265, "D: Evaluación común", "#FFFFFF", "#6F6F6F", header_fill=GREEN_FILL, header_stroke=GREEN_STROKE)
    image(ax, assets["curve"][1], 902, 212, 170, 116)
    image(ax, assets["confusion"][1], 1082, 220, 82, 82)

    arrow(ax, (308, 230), (402, 145))
    arrow(ax, (308, 314), (402, 405))
    arrow(ax, (832, 156), (875, 242))
    arrow(ax, (832, 407), (875, 304))

    fig.savefig(OUT_DIR / "fig_pipeline_experimental.drawio.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def geom(parent: ET.Element, x: float, y: float, w: float, h: float) -> None:
    ET.SubElement(parent, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


class Drawio:
    def __init__(self) -> None:
        self.i = 2
        self.mxfile = ET.Element("mxfile", {"host": "Electron", "version": "29.6.1"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": "experimental-pipeline", "name": "Experimental pipeline"})
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1260",
                "dy": "720",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(WIDTH),
                "pageHeight": str(HEIGHT),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _id(self) -> str:
        value = str(self.i)
        self.i += 1
        return value

    def vertex(self, value: str, style: str, x: float, y: float, w: float, h: float) -> None:
        cell = ET.SubElement(self.root, "mxCell", {"id": self._id(), "value": value, "style": style, "vertex": "1", "parent": "1"})
        geom(cell, x, y, w, h)

    def edge(self, start: tuple[float, float], end: tuple[float, float], *, color: str = INK, width: float = 2.0) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(),
                "value": "",
                "style": f"endArrow=block;html=1;rounded=0;strokeWidth={width};strokeColor={color};",
                "edge": "1",
                "parent": "1",
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geometry, "mxPoint", {"x": str(start[0]), "y": str(start[1]), "as": "sourcePoint"})
        ET.SubElement(geometry, "mxPoint", {"x": str(end[0]), "y": str(end[1]), "as": "targetPoint"})

    def save(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=False)


def xml_value(value: str) -> str:
    return html.escape(value, quote=True)


def box_style(fill: str, stroke: str, *, rounded_box: bool = True, stroke_width: float = 1.0, font_size: float = 10) -> str:
    return (
        f"rounded={1 if rounded_box else 0};whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
        f"fontFamily=Arial;fontSize={font_size};fontColor=#202124;align=center;verticalAlign=middle;spacing=0;"
    )


def image_style() -> str:
    return "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;spacing=0;whiteSpace=wrap;overflow=hidden;"


def image_value(b64: str, w: int, h: int) -> str:
    return f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}">'


def add_card(d: Drawio, x: float, y: float, w: float, h: float, header: str, fill: str, stroke: str, header_fill: str, header_stroke: str) -> None:
    d.vertex("", box_style("#ECEFF3", "#ECEFF3", stroke_width=0), x + 5, y + 7, w, h)
    d.vertex("", box_style(fill, stroke, stroke_width=1.1), x, y, w, h)
    header_w = min(w - 42, max(120, len(header) * 8.4 + 34))
    d.vertex(xml_value(header), box_style(header_fill, header_stroke, font_size=10.8), x + (w - header_w) / 2, y - 18, header_w, 35)


def write_drawio(assets: dict[str, tuple[str, np.ndarray]]) -> None:
    d = Drawio()
    d.edge((308, 230), (402, 145))
    d.edge((308, 314), (402, 405))
    d.edge((832, 156), (875, 242))
    d.edge((832, 407), (875, 304))

    add_card(d, 48, 150, 260, 245, "A: Entrada común", "#FFFFFF", "#8A8A8A", BLUE_FILL, BLUE_STROKE)
    d.vertex("Sintético", box_style(BLUE_FILL, BLUE_STROKE, font_size=8.2), 76, 184, 96, 24)
    d.vertex("HUCA", box_style(GREEN_FILL, GREEN_STROKE, font_size=8.2), 188, 184, 84, 24)
    d.vertex(image_value(assets["synthetic"][0], 103, 100), image_style(), 72, 226, 103, 100)
    d.vertex(image_value(assets["clinical"][0], 96, 100), image_style(), 186, 226, 96, 100)

    add_card(d, 402, 58, 430, 190, "B: CNN", "#FFFFFF", "#6F6F6F", BLUE_FILL, BLUE_STROKE)
    d.vertex(image_value(assets["cnn_pipeline"][0], 390, 135), image_style(), 422, 97, 390, 135)

    add_card(d, 402, 312, 430, 190, "C: VLM/ICL", "#FFFFFF", "#6F6F6F", PURPLE_FILL, PURPLE_STROKE)
    d.vertex(image_value(assets["vlm_pipeline"][0], 390, 135), image_style(), 422, 352, 390, 135)

    add_card(d, 875, 130, 305, 265, "D: Evaluación común", "#FFFFFF", "#6F6F6F", GREEN_FILL, GREEN_STROKE)
    d.vertex(image_value(assets["curve"][0], 170, 116), image_style(), 902, 212, 170, 116)
    d.vertex(image_value(assets["confusion"][0], 82, 82), image_style(), 1082, 220, 82, 82)

    d.save(OUT_DIR / "fig_pipeline_experimental.drawio")


if __name__ == "__main__":
    main()
