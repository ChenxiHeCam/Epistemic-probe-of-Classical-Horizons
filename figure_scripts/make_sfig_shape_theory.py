"""Supplementary figure: shape and mechanism on four measured NIST series.

Rows: mechanism inside the pre-1900 model set (classical or metrological) or
after 1900. Columns: a simple shape or a sharp, non-monotone or abrupt shape.
Scores are the mechanism-group medians from
results/real_measurement_case_evaluation.json (learned score from the compact
encoder; extrapolation score, size-matched).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "figures" / "sfigure_shape_mechanism"

CELLS = [
    # row, col, mechanism group, title, x label, y label, log-x
    (0, 0, "load_cell_elasticity", "Load-cell elasticity", "Load / 10$^{6}$", "Deflection"),
    (0, 1, "circular_interference", "Circular interference", "Wavelength (nm)", "Transmittance"),
    (1, 0, "superconducting_flux_creep", "Magnetization relaxation", "ln(time / min)", "Magnetization"),
    (1, 1, "bose_gas_density", "Cold Bose gas density", "Position", "Density"),
]


def main():
    S.apply()
    d = np.load(ROOT / "data" / "real_measurement_case_pointclouds.npz")
    groups = [str(x) for x in d["mechanism_groups"]]
    scores = json.loads((ROOT / "results" / "real_measurement_case_evaluation.json")
                        .read_text(encoding="utf-8"))["group_scores"]
    fig = plt.figure(figsize=(S.WIDTH, 120 * S.MM))
    gs = fig.add_gridspec(2, 2, left=0.15, right=0.98, top=0.88, bottom=0.08, hspace=0.75, wspace=0.32)
    for row, col, group, title, xl, yl in CELLS:
        ax = fig.add_subplot(gs[row, col])
        cloud = np.asarray(d[f"cloud_{groups.index(group)}"], float)
        if group == "load_cell_elasticity":
            cloud[:, 0] = cloud[:, 0] / 1e6
        post = scores[group]["horizon_class"] == "post_1900_mechanism"
        color = S.LATER if post else S.CLASSICAL
        ax.plot(cloud[:, 0], cloud[:, 1], "s" if post else "o", ms=1.8, color=color,
                mfc=color if post else "white", mew=0.6)
        ax.set_xlabel(xl, fontsize=6.5)
        ax.set_ylabel(yl, fontsize=6.5)
        ax.tick_params(labelsize=6.5)
        g = scores[group]
        rising = col == 0
        ax.text(0.98, 0.04 if rising else 0.96, f"Learned {g['learned_group_score']:.3f}\nExtrapolation {g['statistical_group_score']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom" if rising else "top", fontsize=6.5, color=S.INK,
                bbox=dict(facecolor="white", edgecolor="#BDBDBD", lw=0.5, boxstyle="round,pad=0.3"))
        ax.set_title(title, fontsize=7, loc="left", pad=4)
        S.grid(ax, "both")
    fig.text(0.37, 0.955, "Simple shape", ha="center", fontsize=7.5, fontweight="bold")
    fig.text(0.79, 0.955, "Sharp or abrupt shape", ha="center", fontsize=7.5, fontweight="bold")
    fig.text(0.04, 0.70, "Mechanism in the\npre-1900 model set", ha="center", va="center", rotation=90,
             fontsize=7.5, fontweight="bold", color=S.CLASSICAL, linespacing=1.2)
    fig.text(0.04, 0.27, "Mechanism\nafter 1900", ha="center", va="center", rotation=90, fontsize=7.5,
             fontweight="bold", color=S.LATER, linespacing=1.2)
    S.save(fig, OUT)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
