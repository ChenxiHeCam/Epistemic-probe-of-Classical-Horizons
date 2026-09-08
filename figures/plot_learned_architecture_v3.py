"""Vector-first conceptual schematic for the learned EPOCH v3 component."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "figures" / "figure_learned_architecture_v3.png"
OUT_PDF = ROOT / "figures" / "figure_learned_architecture_v3.pdf"

BLUE = "#0072B2"
PALE_BLUE = "#E8F2F8"
ORANGE = "#D55E00"
PALE_ORANGE = "#FBEDE7"
PURPLE = "#7A5AA6"
PALE_PURPLE = "#F1ECF7"
GREEN = "#009E73"
GREY = "#7F8C8D"
PALE_GREY = "#F0F1F0"
INK = "#252525"
BG = "#FBFBF9"


def box(ax, xy, wh, text, face, edge, fontsize=8.2, weight="normal", radius=0.018):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight=weight, linespacing=1.3)
    return patch


def arrow(ax, start, end, color=GREY, text=None, offset=(0, 0.018), style="-"):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.15, color=color, linestyle=style,
        shrinkA=4, shrinkB=4,
    )
    ax.add_patch(patch)
    if text:
        ax.text((start[0] + end[0]) / 2 + offset[0],
                (start[1] + end[1]) / 2 + offset[1], text,
                ha="center", va="bottom", fontsize=7.1, color=color)


def graph_icon(ax, centre, scale=1.0, color=BLUE):
    cx, cy = centre
    points = np.asarray([
        [cx, cy + 0.045 * scale],
        [cx - 0.045 * scale, cy - 0.005 * scale],
        [cx + 0.045 * scale, cy - 0.005 * scale],
        [cx - 0.07 * scale, cy - 0.055 * scale],
        [cx - 0.02 * scale, cy - 0.055 * scale],
        [cx + 0.045 * scale, cy - 0.055 * scale],
    ])
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    for i, j in edges:
        ax.plot(points[[i, j], 0], points[[i, j], 1], color=color,
                linewidth=1.05, zorder=3)
    for x, y in points:
        ax.add_patch(Circle((x, y), 0.009 * scale, facecolor=BG,
                            edgecolor=color, linewidth=1.0, zorder=4))


def point_icon(ax, origin, width=0.12, height=0.09, color=PURPLE, later=False):
    x0, y0 = origin
    x = np.linspace(0, 1, 12)
    y = 0.15 + 0.66 * x ** (1.5 if not later else 1.05)
    if later:
        y[7:] = y[7] + 0.10 * np.sin(np.linspace(0, np.pi, len(y) - 7))
    ax.scatter(x0 + width * x, y0 + height * y, s=7, color=color,
               linewidth=0, zorder=5)


def panel_a(ax):
    ax.text(0.02, 0.95, "a   Positive-only historical representation learning",
            ha="left", va="top", fontsize=10, fontweight="bold")
    box(ax, (0.04, 0.66), (0.24, 0.17),
        "audited\npre-1900\nformula graphs", PALE_BLUE, BLUE,
        fontsize=7.5, weight="bold")
    graph_icon(ax, (0.16, 0.60), 0.85, BLUE)
    box(ax, (0.37, 0.64), (0.25, 0.21),
        "graph teacher\n4 local updates\n(one RBF-KAN)\n+ 2 attention layers",
        PALE_BLUE, BLUE, fontsize=7.1)
    box(ax, (0.04, 0.25), (0.24, 0.17),
        "normal-law\npoint sets\n(noise + resampling)", PALE_PURPLE, PURPLE,
        fontsize=7.5, weight="bold")
    point_icon(ax, (0.095, 0.16), color=PURPLE)
    box(ax, (0.37, 0.23), (0.25, 0.21),
        "Set Transformer\npoint student\n(query encoder)",
        PALE_PURPLE, PURPLE, fontsize=7.8)
    arrow(ax, (0.28, 0.745), (0.37, 0.745), BLUE)
    arrow(ax, (0.28, 0.335), (0.37, 0.335), PURPLE)
    arrow(ax, (0.50, 0.64), (0.50, 0.44), ORANGE,
          "multi-positive alignment", offset=(0.0, 0.006), style="--")
    box(ax, (0.67, 0.36), (0.30, 0.34),
        "shared 256-d\ngeometry\n\nsame-family\nalignment",
        "#F7F7F5", GREY, fontsize=7.6, weight="bold")
    arrow(ax, (0.62, 0.745), (0.70, 0.62), BLUE)
    arrow(ax, (0.62, 0.335), (0.70, 0.44), PURPLE)
    ax.text(
        0.04, 0.055,
        "No post-cutoff formulas • no saturation/step/kink labels • no units or dates",
        ha="left", va="bottom", fontsize=7.8, color=GREY,
    )


def panel_b(ax):
    ax.text(0.02, 0.95, "b   Frozen one-class audit",
            ha="left", va="top", fontsize=10, fontweight="bold")
    box(ax, (0.04, 0.68), (0.24, 0.16),
        "unordered (x, y)\nquery cloud\nonly input at test",
        PALE_PURPLE, PURPLE, fontsize=7.8, weight="bold")
    point_icon(ax, (0.10, 0.58), color=PURPLE, later=True)
    box(ax, (0.37, 0.66), (0.23, 0.20),
        "frozen\npoint encoder\n(20.9 M model)",
        PALE_PURPLE, PURPLE, fontsize=7.6)
    arrow(ax, (0.28, 0.76), (0.37, 0.76), PURPLE)
    memories = [
        "graph-formula kNN",
        "identity-centroid kNN",
        "point-prototype kNN",
        "prototype Mahalanobis",
    ]
    y_positions = [0.60, 0.47, 0.34, 0.21]
    for label, y in zip(memories, y_positions):
        box(ax, (0.37, y - 0.045), (0.30, 0.075), label,
            "#F7F7F5", GREY, fontsize=7.5, radius=0.012)
        arrow(ax, (0.49, 0.66), (0.49, y + 0.03), GREY)
    box(ax, (0.75, 0.39), (0.20, 0.20),
        "pre-cutoff CDF\nper distance\n+\nequal mean",
        PALE_BLUE, BLUE, fontsize=7.4, weight="bold")
    for y in y_positions:
        arrow(ax, (0.67, y - 0.007), (0.75, 0.46), GREY)
    box(ax, (0.75, 0.14), (0.20, 0.13),
        "one-class\nanomaly score", PALE_ORANGE, ORANGE, weight="bold")
    arrow(ax, (0.85, 0.39), (0.85, 0.27), ORANGE)
    ax.text(0.04, 0.055,
            "cloud → formula → source-law family; primary endpoint: grouped AUROC",
            ha="left", va="bottom", fontsize=7.8, color=GREY)


def panel_c(ax):
    ax.text(0.02, 0.95, "c   Frozen evaluation at the 1899 horizon",
            ha="left", va="top", fontsize=10, fontweight="bold")
    box(ax, (0.04, 0.66), (0.25, 0.16),
        "held-out\npre-1900 controls", PALE_BLUE, BLUE, weight="bold")
    box(ax, (0.04, 0.25), (0.25, 0.16),
        "post-1900\nquery families", PALE_ORANGE, ORANGE, weight="bold")
    point_icon(ax, (0.105, 0.16), color=ORANGE, later=True)
    box(ax, (0.39, 0.46), (0.23, 0.16),
        "same frozen\n1899-horizon\nscore", PALE_PURPLE, PURPLE, weight="bold")
    arrow(ax, (0.29, 0.74), (0.39, 0.57), BLUE)
    arrow(ax, (0.29, 0.33), (0.39, 0.49), ORANGE)
    box(ax, (0.71, 0.62), (0.24, 0.18),
        "family-level\nranking AUROC", "#F7F7F5", GREY, weight="bold")
    box(ax, (0.71, 0.25), (0.24, 0.18),
        "sampling/noise-\nmatched control\ndiagnostic", "#E8F5F0", GREEN,
        fontsize=7.4, weight="bold")
    arrow(ax, (0.62, 0.57), (0.71, 0.70), PURPLE)
    arrow(ax, (0.62, 0.49), (0.71, 0.34), GREEN)
    ax.text(
        0.05, 0.055,
        "Primary: later-family ranking; diagnostic: controls regenerated under the later sampling process",
        ha="left", va="bottom", fontsize=7.8, color=GREY,
    )


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.3), facecolor=BG)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(BG)
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    for x in (0.337, 0.663):
        fig.lines.append(plt.Line2D([x, x], [0.08, 0.90], transform=fig.transFigure,
                                    color="#D9D9D6", linewidth=0.8))
    fig.suptitle(
        "Learning the geometry of an explicit knowledge horizon",
        x=0.035, y=0.985, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(
        0.035, 0.938,
        "Formula structure supervises representation learning only inside the admitted era; "
        "future queries are scored one-class, without a breakdown template.",
        ha="left", va="top", fontsize=8.8, color="#444444",
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.08, wspace=0.08)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=BG)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
