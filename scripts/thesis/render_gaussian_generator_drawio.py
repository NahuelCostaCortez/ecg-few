from __future__ import annotations

import base64
import io
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ecg_few.simulator import generate_beat


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"

WIDTH = 1080
HEIGHT = 560

INK = "#202124"
MUTED = "#5f6368"
BLUE_FILL = "#dbe8fb"
BLUE_STROKE = "#7d94b5"
GREEN_FILL = "#d9ead3"
GREEN_STROKE = "#93c47d"
GRAY_FILL = "#f2f2f2"
GRAY_STROKE = "#b7b7b7"
PURPLE_FILL = "#eadff1"
PURPLE_STROKE = "#9673a6"


def fig_to_b64_and_array(fig: plt.Figure, *, transparent: bool = True) -> tuple[str, np.ndarray]:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.03, transparent=transparent)
    plt.close(fig)
    data = buf.getvalue()
    return base64.b64encode(data).decode("ascii"), plt.imread(io.BytesIO(data))


def formula_image() -> tuple[str, np.ndarray]:
    fig, ax = plt.subplots(figsize=(3.95, 1.20), dpi=180)
    ax.axis("off")
    ax.text(
        0.5,
        0.72,
        r"$g_i(t)=a_i\exp\!\left(-\frac{(t-\mu_i)^2}{2\sigma_i^2}\right)$",
        ha="center",
        va="center",
        fontsize=16,
        color=INK,
    )
    ax.text(
        0.5,
        0.22,
        r"$x(t)=\sum_i g_i(t)$",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
    )
    return fig_to_b64_and_array(fig)


def gaussian(t_ms: np.ndarray, mu: float, sigma: float, a: float) -> np.ndarray:
    return a * np.exp(-((t_ms - mu) ** 2) / (2.0 * sigma**2))


def phase_explanation_image() -> tuple[str, np.ndarray]:
    t_ms = np.linspace(0, 900, 650)
    components = [
        ("P", 125, 28, 0.10, "#5b8ec9"),
        ("Q", 258, 8, -0.03, "#2f75b5"),
        ("R", 305, 12, 0.64, "#2f75b5"),
        ("S", 344, 20, -0.54, "#2f75b5"),
        ("R'", 385, 24, 0.38, "#2f75b5"),
        ("J", 412, 18, 0.22, "#c0392b"),
        ("ST1", 505, 62, 0.24, "#c0392b"),
        ("ST2", 585, 72, -0.03, "#c0392b"),
        ("T1", 690, 78, -0.18, "#4f8f3a"),
        ("T2", 742, 94, -0.04, "#4f8f3a"),
    ]
    curves = [(name, gaussian(t_ms, mu, sigma, amp), color) for name, mu, sigma, amp, color in components]
    waveform = np.sum([curve for _, curve, _ in curves], axis=0)

    fig, ax = plt.subplots(figsize=(6.35, 1.95), dpi=180)
    ax.set_facecolor("#ffffff")
    regions = [
        (80, 180, "#eef3fb", "P", "#5b8ec9"),
        (245, 430, "#dbe8fb", "QRS / RBBB", "#2f75b5"),
        (392, 585, "#fce4d6", "J-ST", "#c0392b"),
        (560, 820, "#d9ead3", "T", "#4f8f3a"),
    ]
    for x0, x1, fill, name, color in regions:
        ax.axvspan(x0, x1, color=fill, alpha=0.72, zorder=0)
        ax.text((x0 + x1) / 2, 0.86, name, fontsize=9, color=color, ha="center", va="center", weight="bold")

    for name, curve, color in curves:
        ax.plot(t_ms, curve, color=color, lw=0.9, alpha=0.46, zorder=2)

    ax.plot(t_ms, waveform, color=INK, lw=2.0, zorder=3)
    ax.axhline(0, color="#8a8a8a", lw=0.9, zorder=2)
    ax.text(22, 0.70, r"$x(t)$", fontsize=14, color=INK, weight="bold")
    ax.text(845, -0.22, r"$t$", fontsize=12, color=MUTED)
    ax.text(735, 0.62, r"$x(t)=\sum_i g_i(t)$", fontsize=10.5, color=INK, ha="center", va="center")

    for x, label_text, color in [
        (125, "P", "#5b8ec9"),
        (305, "R", "#2f75b5"),
        (344, "S", "#2f75b5"),
        (385, "R'", "#2f75b5"),
        (412, "J", "#c0392b"),
        (505, "ST1", "#c0392b"),
        (585, "ST2", "#c0392b"),
        (690, "T1", "#4f8f3a"),
        (742, "T2", "#4f8f3a"),
    ]:
        ax.text(x, -0.72, label_text, fontsize=8.5, color=color, ha="center", va="center", weight="bold")

    ax.set_xlim(0, 900)
    ax.set_ylim(-0.86, 0.98)
    ax.axis("off")
    return fig_to_b64_and_array(fig)


def simulator_example() -> tuple[str, np.ndarray, dict[str, int]]:
    waveform, metadata = generate_beat(
        class_name="BRUGADA",
        seed=2029,
        subtype="coved",
        fs=500,
        duration_ms=900,
    )
    labels = {name: int(value) for name, value in metadata["feature_labels"].items()}
    t_ms = np.arange(waveform.shape[0], dtype=float) / 500.0 * 1000.0

    fig, ax = plt.subplots(figsize=(3.7, 1.0), dpi=180)
    ax.set_facecolor("#fffafa")
    for x in np.arange(0, 901, 20):
        ax.axvline(x, color="#f6d4d4", lw=0.4, zorder=0)
    for x in np.arange(0, 901, 100):
        ax.axvline(x, color="#ecaaaa", lw=0.8, zorder=0)
    for y in np.arange(-1.2, 1.1, 0.1):
        ax.axhline(y, color="#f6d4d4", lw=0.4, zorder=0)
    for y in np.arange(-1.0, 1.1, 0.5):
        ax.axhline(y, color="#ecaaaa", lw=0.8, zorder=0)
    ax.plot(t_ms, waveform, color=INK, lw=1.45, zorder=3)
    ax.set_xlim(0, 900)
    ax.set_ylim(-1.05, 0.95)
    ax.axis("off")
    ecg_b64, ecg_img = fig_to_b64_and_array(fig, transparent=False)
    return ecg_b64, ecg_img, labels


def rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, fill: str, stroke: str, lw: float = 1.1, radius: float = 7) -> None:
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


def rect(ax: plt.Axes, x: float, y: float, w: float, h: float, fill: str, stroke: str, lw: float = 1.0) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke, linewidth=lw))


def label(ax: plt.Axes, x: float, y: float, value: str, **kwargs) -> None:
    opts = {
        "ha": "center",
        "va": "center",
        "fontfamily": "Arial",
        "fontsize": 10,
        "color": INK,
    }
    opts.update(kwargs)
    ax.text(x, y, value, **opts)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, lw: float = 1.9) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=INK,
            shrinkA=0,
            shrinkB=0,
        )
    )


def parameter_card(ax: plt.Axes, x: float, header: str, body: str) -> None:
    rounded(ax, x + 4, 59, 205, 92, "#efefef", "#efefef", lw=0.0, radius=6)
    rounded(ax, x, 55, 205, 92, "#ffffff", "#8a8a8a", lw=1.0, radius=6)
    rounded(ax, x + 30, 37, 145, 31, BLUE_FILL, BLUE_STROKE, lw=1.0, radius=3)
    label(ax, x + 102.5, 52.5, header, fontsize=9.4, fontweight="bold")
    rounded(ax, x + 42, 88, 121, 27, GRAY_FILL, GRAY_STROKE, lw=0.9, radius=2)
    label(ax, x + 102.5, 101.5, body, fontsize=11)


def draw_preview(phase_img: np.ndarray, formula_img: np.ndarray, ecg_img: np.ndarray, labels: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    parameter_card(ax, 95, "A: Amplitud", r"$a_i$")
    parameter_card(ax, 360, r"$\mu$: Centro", r"$\mu_i$")
    parameter_card(ax, 625, r"$\sigma$: Anchura", r"$\sigma_i$")

    arrow(ax, (197.5, 147), (360, 238), lw=1.5)
    arrow(ax, (462.5, 147), (462.5, 238), lw=1.9)
    arrow(ax, (727.5, 147), (565, 238), lw=1.5)

    rounded(ax, 176, 246, 585, 182, "#eceff3", "#eceff3", lw=0.0, radius=14)
    rounded(ax, 170, 238, 585, 182, "#ffffff", "#6f6f6f", lw=1.2, radius=14)
    rounded(ax, 365, 220, 195, 31, BLUE_FILL, BLUE_STROKE, lw=1.0, radius=4)
    label(ax, 462.5, 235.5, "B: Generador QRS/ST", fontsize=9.2, fontweight="bold")
    ax.imshow(phase_img, extent=(215, 710, 392, 268), aspect="auto", zorder=3)

    arrow(ax, (755, 329), (835, 329))

    rounded(ax, 840, 250, 225, 160, "#efefef", "#efefef", lw=0.0, radius=10)
    rounded(ax, 835, 245, 225, 160, "#ffffff", "#8a8a8a", lw=1.0, radius=10)
    rounded(ax, 855, 227, 185, 31, GREEN_FILL, GREEN_STROKE, lw=1.0, radius=3)
    label(ax, 947.5, 242.5, "A: Ejemplo simulado", fontsize=9.0, fontweight="bold")
    ax.imshow(ecg_img, extent=(865, 1030, 318, 268), aspect="auto", zorder=3)
    label(
        ax,
        947.5,
        372,
        f"RBBB: {'SÍ' if labels['RBBB'] else 'NO'}  ST: {'SÍ' if labels['ST_ELEVATION'] else 'NO'}  TWI: {'SÍ' if labels['T_WAVE_INVERSION'] else 'NO'}",
        fontsize=9.1,
        fontweight="bold",
    )

    arrow(ax, (482.5, 450), (482.5, 420), lw=1.4)
    rounded(ax, 340, 452, 285, 106, PURPLE_FILL, PURPLE_STROKE, lw=1.5, radius=9)
    ax.imshow(formula_img, extent=(365, 600, 540, 468), aspect="auto", zorder=3)

    fig.savefig(OUT_DIR / "gaussian_generator.drawio.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def geom(parent: ET.Element, x: float, y: float, w: float, h: float) -> None:
    ET.SubElement(parent, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


class Drawio:
    def __init__(self) -> None:
        self.i = 2
        self.mxfile = ET.Element("mxfile", {"host": "Electron", "version": "29.6.1"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": "gaussian-generator-sketch", "name": "Gaussian generator sketch"})
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1188",
                "dy": "708",
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

    def edge(self, start: tuple[float, float], end: tuple[float, float], *, width: float = 2.0) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(),
                "value": "",
                "style": f"endArrow=block;html=1;rounded=0;strokeWidth={width};strokeColor=#202124;",
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


def box_style(fill: str, stroke: str, *, rounded_box: bool = True, stroke_width: float = 1.0) -> str:
    return (
        f"rounded={1 if rounded_box else 0};whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
        "fontFamily=Arial;fontSize=10;fontColor=#202124;align=center;verticalAlign=middle;spacing=0;"
    )


def image_style() -> str:
    return "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;spacing=0;whiteSpace=wrap;overflow=hidden;"


def image_value(b64: str, w: int, h: int) -> str:
    return f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}">'


def write_drawio(phase_b64: str, formula_b64: str, ecg_b64: str, labels: dict[str, int]) -> None:
    d = Drawio()
    d.edge((197.5, 147), (360, 238), width=1.5)
    d.edge((462.5, 147), (462.5, 238), width=2.0)
    d.edge((727.5, 147), (565, 238), width=1.5)
    d.edge((755, 329), (835, 329), width=2.0)
    d.edge((482.5, 450), (482.5, 420), width=1.4)

    for x, header, body in [
        (95, "A: Amplitud", "a_i"),
        (360, "μ: Centro", "μ_i"),
        (625, "σ: Anchura", "σ_i"),
    ]:
        d.vertex("", box_style("#ffffff", "#8a8a8a"), x, 55, 205, 92)
        d.vertex(header, box_style(BLUE_FILL, BLUE_STROKE), x + 30, 37, 145, 31)
        d.vertex(body, box_style(GRAY_FILL, GRAY_STROKE, rounded_box=True, stroke_width=0.9), x + 42, 88, 121, 27)

    d.vertex(
        "",
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#6f6f6f;strokeWidth=1.2;shadow=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#202124;align=center;verticalAlign=middle;spacing=0;",
        170,
        238,
        585,
        182,
    )
    d.vertex("B: Generador QRS/ST", box_style(BLUE_FILL, BLUE_STROKE), 365, 220, 195, 31)
    d.vertex(image_value(phase_b64, 495, 124), image_style(), 215, 268, 495, 124)

    d.vertex(
        "",
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#8a8a8a;strokeWidth=1;shadow=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#202124;align=center;verticalAlign=middle;spacing=0;",
        835,
        245,
        225,
        160,
    )
    d.vertex("A: Ejemplo simulado", box_style(GREEN_FILL, GREEN_STROKE), 855, 227, 185, 31)
    d.vertex(image_value(ecg_b64, 165, 50), image_style(), 865, 268, 165, 50)
    d.vertex(
        f"RBBB: {'SÍ' if labels['RBBB'] else 'NO'}  ST: {'SÍ' if labels['ST_ELEVATION'] else 'NO'}  TWI: {'SÍ' if labels['T_WAVE_INVERSION'] else 'NO'}",
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;fontFamily=Arial;fontSize=10;fontStyle=1;fontColor=#202124;align=center;verticalAlign=middle;spacing=0;",
        865,
        358,
        165,
        24,
    )

    d.vertex("", box_style(PURPLE_FILL, PURPLE_STROKE, stroke_width=1.5), 340, 452, 285, 106)
    d.vertex(image_value(formula_b64, 235, 72), image_style(), 365, 468, 235, 72)

    d.save(OUT_DIR / "gaussian_generator.drawio")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_b64, phase_img = phase_explanation_image()
    formula_b64, formula_img = formula_image()
    ecg_b64, ecg_img, labels = simulator_example()
    write_drawio(phase_b64, formula_b64, ecg_b64, labels)
    draw_preview(phase_img, formula_img, ecg_img, labels)
    print(OUT_DIR / "gaussian_generator.drawio")
    print(OUT_DIR / "gaussian_generator.drawio.png")


if __name__ == "__main__":
    main()
