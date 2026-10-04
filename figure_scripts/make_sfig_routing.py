"""Supplementary figure: the rule that assigns a point cloud to a test or screen.

The extrapolation screen ends in a descriptive classification (compatible or
localized incompatibility); only the named-prediction test issues a verdict.
When a fitted regime exists and a pre-1900 prediction and error model are also
supplied, both are run and reported (dashed arrow). Boxes and arrows reuse the
helpers of make_fig1_schematic.py; colours follow epoch_style.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import make_fig1_schematic as M  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "figures" / "sfigure_routing"


def flow(ax):
    box, arrow = M.box, M.arrow
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.00, 0.70, 0.20, 0.27, "Measurements", "(x, y) pairs; pre-1900\nprediction and error\nmodel if supplied")
    box(ax, 0.27, 0.72, 0.21, 0.24, "Does an admitted\npre-1900 form fit a\ncontiguous fitted\nregime?")
    box(ax, 0.55, 0.60, 0.26, 0.38, "Extrapolation screen",
        "Fit a power, exponential,\nlinear or constant form\nin the fitted regime;\nextrapolate; score the\nresiduals in the\nextrapolation region",
        edge=S.FOURFORM, lw=1.2)
    box(ax, 0.85, 0.60, 0.16, 0.38, "Descriptive\nclassification",
        "Compatible over\nthe measured\nrange\n\nLocalized\nincompatibility,\nwith boundary\nestimate")
    box(ax, 0.27, 0.22, 0.21, 0.22, "Pre-1900 prediction\nand error model\ngiven?")
    box(ax, 0.55, 0.12, 0.26, 0.34, "Named-prediction\ntest",
        "Residuals from the\npre-1900 prediction\nagainst a null simulated\non the same grid", edge=S.INK, lw=1.2)
    box(ax, 0.85, 0.10, 0.16, 0.38, "Verdict",
        "Compatible over\nthe measured\nrange\n\nLocalized\nincompatibility\n\nWhole-range\nincompatibility",
        lw=1.2)
    box(ax, 0.27, 0.01, 0.21, 0.12, "Abstention")
    box(ax, 0.00, 0.16, 0.21, 0.42, "Representation\nscreen",
        "Encoder trained\non pre-1900\nrelations only;\nscores point clouds\nwithin its operating\nrange; reported\nseparately",
        edge=S.LEARNED, lw=1.2)
    arrow(ax, (0.20, 0.84), (0.27, 0.84))
    arrow(ax, (0.48, 0.84), (0.55, 0.84), "yes", ty=0.03)
    arrow(ax, (0.375, 0.72), (0.375, 0.44), "no", tx=0.03)
    arrow(ax, (0.48, 0.33), (0.55, 0.33), "yes", ty=0.03)
    arrow(ax, (0.375, 0.22), (0.375, 0.13), "no", tx=0.03)
    arrow(ax, (0.81, 0.79), (0.85, 0.79))
    arrow(ax, (0.81, 0.29), (0.85, 0.29))
    arrow(ax, (0.68, 0.60), (0.68, 0.46), ls=(0, (3, 2)))
    ax.text(0.665, 0.53, "Prediction also\nsupplied: both run\nand reported", fontsize=6.5, color=S.MUTED,
            ha="right", va="center", linespacing=1.15, zorder=3)
    arrow(ax, (0.10, 0.70), (0.10, 0.58), color=S.LEARNED, ls=(0, (3, 2)))


def main():
    S.apply()
    fig = plt.figure(figsize=(S.WIDTH, 110 * S.MM))
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    flow(ax)
    S.save(fig, OUT)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
