"""Shared style for the EPOCH manuscript figures.

One colour per meaning, used the same way in every figure:

    CLASSICAL  blue        pre-1900 laws, classical controls, pre-1900 predictions
    LATER      vermillion  post-1900 laws, later physics, successor predictions
    LEARNED    purple      the trained encoders (representation screen)
    FOURFORM   green       the extrapolation screen
    UNTRAINED  grey        untrained network and random or generic controls
    DATA       near-black  measurements and generated points

Palette checked with the dataviz validator (light mode, adjacent and all pairs).
Blue and purple are close for deuteranopes, so wherever both appear they also
differ in line style or marker.

Figures are drawn at the NeurIPS text width (5.5 in) and included at
\textwidth, so font sizes are true printed sizes: 7.5 pt body, 7 pt ticks and
legends, 6.5 pt minimum, 9 pt bold panel letters.

Panel letters are lower case by default. Set the environment variable
EPOCH_PANEL_CASE=upper for upper-case panel letters.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTH = 5.5  # inches: the NeurIPS text width, so figures print at 100%

CLASSICAL = "#0072B2"
LATER = "#D55E00"
LEARNED = "#7B3294"
FOURFORM = "#009E73"
UNTRAINED = "#8C8C8C"
DATA = "#222222"
INK = "#222222"
MUTED = "#5F5F5F"
GRID = "#E6E6E6"
LIGHT = "#D9D9D9"
BG = "white"

UPPER = os.environ.get("EPOCH_PANEL_CASE", "lower").lower() == "upper"


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "lines.linewidth": 1.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def letter(s: str) -> str:
    return s.upper() if UPPER else s


def panel(ax, s: str, title: str | None = None, x: float = -0.02, y: float = 1.04) -> None:
    """Bold panel letter at the top left, with an optional plain descriptive title."""
    ax.text(x, y, letter(s), transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", clip_on=False)
    if title:
        ax.annotate(title, xy=(x, y), xycoords="axes fraction", xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, va="bottom", ha="left",
                    annotation_clip=False)


def grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem, dpi: int = 600) -> None:
    """Write PNG, PDF and SVG next to each other; no bbox cropping so width stays 180 mm."""
    stem = str(stem)
    fig.savefig(stem + ".png", dpi=dpi)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
