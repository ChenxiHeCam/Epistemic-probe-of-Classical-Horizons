"""Figure 1: testing a knowledge horizon (schematic).

a, the knowledge horizon as a date: relations recorded before it are admitted
   (training and pre-1900 predictions), later physics is used for evaluation only.
b, the fitted regime, the extrapolation region and the boundary estimate on one relation.
c, the fixed rule that assigns a point cloud to a test, and the verdicts.

Terms are those of the manuscript. Colours follow figure_scripts/epoch_style.py.
The curve in b is illustrative, not data.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = ROOT / "figures" / "Figure_1_schematic"
DARK = "#4A4A4A"


def box(ax, x, y, w, h, title, body=None, edge=DARK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor="white", zorder=2, clip_on=False))
    if body:
        n = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=7,
                fontweight="bold", color=S.INK, zorder=3, linespacing=1.1)
        ax.text(x + w / 2, y + h - 0.04 - 0.05 * n, body, ha="center", va="top", fontsize=6.5,
                color=S.INK, linespacing=1.25, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=6.5,
                color=S.INK, linespacing=1.25, zorder=3)


def arrow(ax, p, q, text=None, color=DARK, ls="-", tx=0.0, ty=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=2, zorder=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=6.5, color=S.MUTED,
                ha="center", va="center", zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))


def T(yr):
    """Piecewise year scale: 1600-1900 over 0-0.6, 1900-2000 over 0.6-1.0."""
    return 0.6 * (yr - 1600) / 300 if yr <= 1900 else 0.6 + 0.4 * (yr - 1900) / 100


def timeline(ax):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.75, 1.2)
    ax.axis("off")
    H = T(1899.5)
    ax.fill_between([0, H], -0.08, 0.08, color=S.CLASSICAL, alpha=0.25, lw=0)
    ax.fill_between([H, 1.0], -0.08, 0.08, color=S.LATER, alpha=0.22, lw=0)
    for yr in (1600, 1650, 1700, 1750, 1800, 1850, 1900, 1925, 1950, 1975, 2000):
        ax.plot([T(yr)] * 2, [-0.08, -0.16], color=S.INK, lw=0.5)
        ax.text(T(yr), -0.22, str(yr), ha="center", va="top", fontsize=6.5)
    ax.text(0.6, -0.62, "Scale changes at 1900", fontsize=6.5, color=S.MUTED, ha="center", va="top")
    ax.plot([H, H], [-0.5, 1.1], color=S.INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(H + 0.006, 1.1, "Knowledge horizon: 1899", fontsize=7, fontweight="bold", va="top", ha="left")
    ax.text(0.0, 1.1, "Admitted: relations recorded up to 1899\n(training corpus and pre-1900 predictions)",
            fontsize=6.5, va="top", ha="left", color=S.CLASSICAL, linespacing=1.2)
    ax.text(1.0, 0.82, "Later physics: evaluation only", fontsize=6.5, va="top", ha="right", color=S.LATER)
    for yr, name in [(1619, "Kepler"), (1662, "Boyle"), (1819, "Dulong–Petit")]:
        ax.plot(T(yr), 0, "o", ms=3.2, color=S.CLASSICAL, zorder=3)
        ax.text(T(yr), 0.16, f"{name}\n{yr}", fontsize=6.5, ha="center", va="bottom", color=S.CLASSICAL,
                linespacing=1.1)
    for yr, name in [(1905, "Photoelectric\neffect"), (1937, "Cherenkov\nemission"),
                     (1961, "Fano\nresonance"), (1998, "Cosmic\nacceleration")]:
        ax.plot(T(yr), 0, "s", ms=3.0, color=S.LATER, zorder=3)
        left = yr == 1905
        ax.text(T(yr) - (0.008 if left else 0), 0.16, f"{name}\n{yr}", fontsize=6.5,
                ha="left" if left else "center", va="bottom", color=S.LATER, linespacing=1.1)
    S.panel(ax, "a", "The knowledge horizon", x=0.0, y=1.02)


def curve(ax):
    rng = np.random.default_rng(4)
    x = np.linspace(0.05, 1.0, 26)
    xb = 0.58
    incumbent = 0.15 + 1.05 * x
    y = np.where(x < xb, incumbent, incumbent - 2.4 * np.clip(x - xb, 0, None) ** 1.6)
    y = y + rng.normal(0, 0.012, x.size)
    xx = np.linspace(0.03, 1.02, 200)
    ax.axvspan(xb, 1.04, color=S.UNTRAINED, alpha=0.10, lw=0)
    ax.plot(xx[xx <= xb], 0.15 + 1.05 * xx[xx <= xb], color=S.CLASSICAL, lw=1.4)
    ax.plot(xx[xx >= xb], 0.15 + 1.05 * xx[xx >= xb], color=S.CLASSICAL, lw=1.4, ls=(0, (3, 2)))
    ax.scatter(x, y, s=7, color=S.DATA, zorder=3)
    ax.axvline(xb, color=S.INK, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(0.30, 0.08, "Fitted regime", ha="center", fontsize=6.5, fontweight="bold")
    ax.text(0.81, 0.08, "Extrapolation\nregion", ha="center", fontsize=6.5, fontweight="bold", linespacing=1.1)
    ax.text(0.27, 0.64, "Pre-1900 law fitted\nin this regime", fontsize=6.5, color=S.CLASSICAL, ha="center",
            linespacing=1.2)
    ax.text(1.0, 1.25, "Extrapolation", fontsize=6.5, color=S.CLASSICAL, ha="right")
    ax.text(xb - 0.015, 1.31, "Boundary\nestimate", fontsize=6.5, ha="right", va="top", color=S.INK,
            linespacing=1.15)
    xa = 0.93
    ya = float(np.interp(xa, x, y))
    ax.annotate("", xy=(xa, ya + 0.02), xytext=(xa, 0.15 + 1.05 * xa - 0.02),
                arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=0.6))
    ax.text(xa + 0.02, (ya + 0.15 + 1.05 * xa) / 2, "Residual", fontsize=6.5, color=S.MUTED,
            ha="left", va="center", linespacing=1.15)
    ax.set_xlim(0.0, 1.12)
    ax.set_ylim(0.0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    S.panel(ax, "b", "Fitted regime and extrapolation region", x=0.0)


def flow(ax):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.00, 0.70, 0.20, 0.27, "Measurements", "(x, y) pairs; pre-1900\nprediction and error\nmodel if supplied")
    box(ax, 0.27, 0.71, 0.21, 0.24, "Does an admitted\npre-1900 form fit a\ncontiguous fitted\nregime?")
    box(ax, 0.55, 0.60, 0.26, 0.38, "Extrapolation screen",
        "Fit a power, exponential,\nlinear or constant form\nin the fitted regime;\nextrapolate; score the\nresiduals in the\nextrapolation region",
        edge=S.FOURFORM, lw=1.2)
    box(ax, 0.27, 0.33, 0.21, 0.24, "Pre-1900 prediction\nand error model\ngiven?")
    box(ax, 0.55, 0.20, 0.26, 0.36, "Named-prediction\ntest",
        "Residuals from the\npre-1900 prediction\nagainst a null simulated\non the same grid", edge=S.INK, lw=1.2)
    box(ax, 0.27, 0.03, 0.21, 0.15, "Abstention")
    box(ax, 0.85, 0.16, 0.16, 0.80, "Verdict",
        "Compatible over\nthe measured\nrange\n\nLocalized\nincompatibility,\nwith boundary\nestimate\n\nWhole-range\nincompatibility,\nno identifiable\nboundary",
        lw=1.2)
    box(ax, 0.00, 0.18, 0.21, 0.41, "Representation\nscreen",
        "Encoder trained\non pre-1900\nrelations only;\nscores point clouds\nwithin its operating\nrange; reported\nseparately",
        edge=S.LEARNED, lw=1.2)
    arrow(ax, (0.20, 0.82), (0.27, 0.82))
    arrow(ax, (0.48, 0.82), (0.55, 0.82), "yes", ty=0.035)
    arrow(ax, (0.375, 0.71), (0.375, 0.57), "no", tx=0.03)
    arrow(ax, (0.48, 0.45), (0.55, 0.45), "yes", ty=0.035)
    arrow(ax, (0.375, 0.33), (0.375, 0.18), "no", tx=0.03)
    arrow(ax, (0.81, 0.79), (0.85, 0.74))
    arrow(ax, (0.81, 0.38), (0.85, 0.44))
    arrow(ax, (0.10, 0.70), (0.10, 0.59), color=S.LEARNED, ls=(0, (3, 2)))
    S.panel(ax, "c", "Which test gives the verdict", x=0.0, y=1.01)


def main():
    S.apply()
    fig = plt.figure(figsize=(S.WIDTH, 106 * S.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1.0], width_ratios=[0.78, 1.22], left=0.035,
                          right=0.985, top=0.95, bottom=0.04, hspace=0.30, wspace=0.12)
    timeline(fig.add_subplot(gs[0, :]))
    curve(fig.add_subplot(gs[1, 0]))
    flow(fig.add_subplot(gs[1, 1]))
    S.save(fig, OUT)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
