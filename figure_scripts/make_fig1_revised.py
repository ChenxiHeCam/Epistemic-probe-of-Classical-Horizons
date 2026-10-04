"""Academic concept figure for the operational, unified EPOCH definition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#6B7280"
LIGHT_GREY = "#F1F3F5"
LIGHT_BLUE = "#EAF4FA"
LIGHT_ORANGE = "#FCEFE9"
INK = "#171717"
BG = "#FBFBF9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def box(ax, xy, width, height, text, edge=GREY, face="white", fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize, weight=weight,
        color=INK, transform=ax.transAxes, linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color=GREY, style="-|>", lw=1.2, dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, transform=ax.transAxes,
        linestyle="--" if dashed else "-", connectionstyle="arc3,rad=0",
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


fig = plt.figure(figsize=(15, 8.2), facecolor=BG)
gs = fig.add_gridspec(2, 2, height_ratios=[1.03, 0.97], width_ratios=[1.03, 1.17], hspace=0.48, wspace=0.25)

# Panel a: operational claim.
ax = fig.add_subplot(gs[0, 0], facecolor=BG)
x = np.linspace(0.04, 0.96, 90)
known = 0.18 + 0.92 * x - 0.22 * x**2
departure = known.copy()
mask = x > 0.56
departure[mask] = known[mask] - 2.5 * (x[mask] - 0.56) ** 1.45

ax.axvspan(0, 0.56, color=LIGHT_BLUE, zorder=0)
ax.axvspan(0.56, 1, color=LIGHT_GREY, zorder=0)
left = x <= 0.56
ax.plot(x[left], known[left], color=BLUE, lw=2.4, zorder=3)
ax.plot(x[~left], known[~left], color=BLUE, lw=2.0, ls=(0, (4, 3)), zorder=2)
rng = np.random.default_rng(4)
idx = np.arange(2, 88, 4)
ax.scatter(x[idx], departure[idx] + rng.normal(0, 0.009, len(idx)), s=18, color=INK, zorder=4)
ax.axvline(0.56, color=ORANGE, lw=1.7, ls=(0, (4, 3)))
ax.scatter([0.56], [0], s=38, color=ORANGE, marker="o", zorder=5, clip_on=False)
ax.fill_between(x[~left], departure[~left], known[~left], color=ORANGE, alpha=0.12, hatch="///", edgecolor=ORANGE, linewidth=0)
ax.text(0.07, 0.93, "COMPATIBLE WITH DECLARED MODELS", color=BLUE, weight="bold", transform=ax.transAxes, fontsize=7.8)
ax.text(0.66, 0.93, "MODEL-INCOMPATIBLE", color=GREY, weight="bold", transform=ax.transAxes, fontsize=8.3)
ax.text(0.575, 0.08, "boundary, if identifiable", color=ORANGE, fontsize=8)
ax.annotate("incumbent extrapolation", xy=(0.81, known[np.argmin(abs(x - 0.81))]), xytext=(0.67, 0.80),
            color=BLUE, fontsize=8, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.annotate("measured departure", xy=(0.82, departure[np.argmin(abs(x - 0.82))]), xytext=(0.69, 0.37),
            color=ORANGE, fontsize=8, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.08)
ax.set_xlabel("frontier variable")
ax.set_ylabel("response")
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("a   Boundary relative to declared models and error process", loc="left", weight="bold", pad=12, fontsize=10.5)

# Panel b: routing, spanning full width at bottom.
ax = fig.add_subplot(gs[1, :], facecolor=BG)
ax.set_axis_off()
ax.set_title("b   A frozen router selects the appropriate internal evidence component", loc="left", weight="bold", pad=10)

box(ax, (0.01, 0.55), 0.15, 0.22, "Measurements\n+ horizon  Hₜ", edge=INK, face="white", weight="bold")
box(ax, (0.205, 0.55), 0.18, 0.22, "Clean incumbent\nregime observed?", edge=GREY, face=LIGHT_GREY, weight="bold")
box(ax, (0.44, 0.67), 0.18, 0.25, "DATA-ANCHORED\nCOMPONENT\nfit the measured interior", edge=BLUE, face=LIGHT_BLUE, weight="bold")
box(ax, (0.44, 0.27), 0.18, 0.25, "Incumbent prediction\n+ measurement model\navailable?", edge=GREY, face=LIGHT_GREY, weight="bold")
box(ax, (0.68, 0.27), 0.16, 0.25, "THEORY-CONDITIONED\nCOMPONENT\ntest a case-matched null", edge=ORANGE, face=LIGHT_ORANGE, weight="bold")
box(ax, (0.68, 0.02), 0.16, 0.13, "ABSTAIN", edge=GREY, face="white", weight="bold")
box(ax, (0.88, 0.47), 0.11, 0.34,
    "SHARED VERDICT\ncompatible\nlocalized mismatch\n+ interval\nwhole-range reject\nabstain",
    edge=INK, face="white", fontsize=6.9, weight="bold")

arrow(ax, (0.16, 0.66), (0.205, 0.66), color=INK)
arrow(ax, (0.385, 0.69), (0.44, 0.79), color=BLUE)
arrow(ax, (0.385, 0.59), (0.44, 0.40), color=GREY)
arrow(ax, (0.62, 0.395), (0.68, 0.395), color=ORANGE)
arrow(ax, (0.53, 0.27), (0.68, 0.10), color=GREY)
arrow(ax, (0.62, 0.795), (0.88, 0.69), color=BLUE)
arrow(ax, (0.84, 0.395), (0.88, 0.59), color=ORANGE)
arrow(ax, (0.84, 0.085), (0.88, 0.50), color=GREY)
ax.text(0.400, 0.76, "yes", color=BLUE, fontsize=8, transform=ax.transAxes)
ax.text(0.400, 0.47, "no", color=GREY, fontsize=8, transform=ax.transAxes)
ax.text(0.642, 0.42, "yes", color=ORANGE, fontsize=8, transform=ax.transAxes)
ax.text(0.605, 0.18, "no", color=GREY, fontsize=8, transform=ax.transAxes)

# Panel c inset: falsifiable clock-advance experiment.
axc = fig.add_subplot(gs[0, 1], facecolor=BG)
axc.set_axis_off()
axc.set_title("c   Advancing the clock: a paired, falsifiable test", loc="left", weight="bold", pad=12, fontsize=10.5)
box(axc, (0.02, 0.56), 0.26, 0.24, "same measurements", edge=INK, face="white", weight="bold")
box(axc, (0.38, 0.66), 0.25, 0.22, "old horizon  Hₜ\nincumbent vocabulary", edge=BLUE, face=LIGHT_BLUE, weight="bold")
box(axc, (0.38, 0.30), 0.25, 0.22, "advanced horizon  Hₜ₊₁\nadds the successor law", edge=ORANGE, face=LIGHT_ORANGE, weight="bold")
box(axc, (0.72, 0.66), 0.25, 0.22, "large calibrated\nmismatch", edge=ORANGE, face="white", weight="bold")
box(axc, (0.72, 0.30), 0.25, 0.22, "mismatch collapses\ntoward the null", edge=BLUE, face="white", weight="bold")
arrow(axc, (0.28, 0.68), (0.38, 0.77), color=BLUE)
arrow(axc, (0.28, 0.66), (0.38, 0.41), color=ORANGE)
arrow(axc, (0.63, 0.77), (0.72, 0.77), color=ORANGE)
arrow(axc, (0.63, 0.41), (0.72, 0.41), color=BLUE)
arrow(axc, (0.845, 0.65), (0.845, 0.53), color=GREY, dashed=True)
axc.text(0.67, 0.08,
         "Required evidence: the score falls for the same data\n"
         "when, and only when, the knowledge horizon is updated.",
         ha="center", va="center", transform=axc.transAxes, color=GREY, fontsize=8.5)
axc.text(0.50, 0.57, "paired test", transform=axc.transAxes, ha="center", color=INK, fontsize=8, weight="bold")

fig.suptitle("EPOCH: auditing the predictive boundary of declared incumbent models", x=0.05, y=0.985,
             ha="left", fontsize=16, weight="bold", color=INK)
fig.text(0.05, 0.945,
         "One method, internally routed: mismatch is not by itself evidence of new physics or a unique successor law.",
         ha="left", fontsize=10, color=GREY)

fig.savefig(OUT / "Figure_1_concept_revised.png", dpi=240, bbox_inches="tight", facecolor=BG)
fig.savefig(OUT / "Figure_1_concept_revised.pdf", bbox_inches="tight", facecolor=BG)
MANUSCRIPT_OUT = ROOT / "manuscript" / "figures"
MANUSCRIPT_OUT.mkdir(exist_ok=True)
fig.savefig(MANUSCRIPT_OUT / "Figure_1_concept_revised.png", dpi=240, bbox_inches="tight", facecolor=BG)
print(OUT / "Figure_1_concept_revised.png")
