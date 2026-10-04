"""Supplementary figure: data design of the large encoder (sfigure_data_design).

What is admitted (relations dated up to 1899), how it is split by law family into training,
calibration and test formulas, how the pre-1900 control groups are built, and which sets are used
only for evaluation. Every count is read from the result files:

  results/phase_a_pre1900_v3_evaluation.json          training relations, law groups, calibration
                                                     formulas, control groups, later families
  results/matched_pre1900_domain_control_trained_v3.json  regenerated control groups
  results/real_measurement_case_evaluation.json       measured series and mechanism groups
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "figures" / "sfigure_data_design"
DARK = "#4A4A4A"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def box(ax, x, y, w, h, head, body, edge=DARK, lw=0.9):
    """Box with a bold heading 4 pt below its top edge and the body 4 pt below the heading."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor="white", zorder=2, clip_on=False))
    nhead = len(head.splitlines())
    ax.annotate(head, (x + w / 2, y + h), xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                fontsize=7, fontweight="bold", linespacing=1.1, zorder=3)
    ax.annotate(body, (x + w / 2, y + h), xytext=(0, -4 - 8.5 * nhead - 4), textcoords="offset points",
                ha="center", va="top", fontsize=6.5, linespacing=1.2, zorder=3)


def arrow(ax, p, q, text=None, tx=0.0, ty=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.8, color=DARK,
                                 shrinkA=0, shrinkB=0, zorder=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=6.5, color=S.MUTED,
                ha=ha, va="center")


def main():
    S.apply()
    ev = load("phase_a_pre1900_v3_evaluation.json")
    matched = load("matched_pre1900_domain_control_trained_v3.json")
    measured = load("real_measurement_case_evaluation.json")["design"]
    assert ev["knowledge_cutoff"] == 1899
    n_train = ev["memory"]["known_graph_formula_embeddings"]
    n_groups = ev["memory"]["known_graph_identity_embeddings"]
    n_cal = ev["calibration_formulas"]
    n_ctrl = len(ev["sets"]["known_internal"]["scores"])
    n_regen = len(matched["sets"]["generator_matched_pre1900_controls"]["scores"])
    n_early = len(ev["sets"]["curated_1901_1950"]["scores"])
    n_late = len(ev["sets"]["curated_post1950"]["scores"])
    n_all = len(ev["sets"]["curated_all_post1900"]["scores"])
    assert n_all == n_early + n_late

    fig = plt.figure(figsize=(S.WIDTH, 70 * S.MM))
    ax = fig.add_axes([0.01, 0.015, 0.98, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.27, 1.03, f"Admitted: relations dated up to {ev['knowledge_cutoff']}", ha="center",
            va="bottom", fontsize=7.5, fontweight="bold", color=S.CLASSICAL)
    ax.text(0.81, 1.03, "Used only for evaluation", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color=S.LATER)

    box(ax, 0.0, 0.80, 0.54, 0.2, "Pre-1900 corpus",
        "dimensionless reductions y = f(x) of pre-1900 laws,\nsampled as point clouds", edge=S.CLASSICAL)
    arrow(ax, (0.27, 0.80), (0.27, 0.73), "split by law family", tx=0.012, ha="left")
    w3 = 0.17
    gap = (0.54 - 3 * w3) / 2
    box(ax, 0.0, 0.42, w3, 0.31, "Training",
        f"{n_train:,} relations\nin {n_groups} law groups", edge=S.CLASSICAL)
    box(ax, w3 + gap, 0.42, w3, 0.31, "Calibration",
        f"{n_cal} held-out\nformulas", edge=S.CLASSICAL)
    box(ax, 2 * (w3 + gap), 0.42, w3, 0.31, "Test",
        "held-out\nformulas", edge=S.CLASSICAL)
    xt = 2 * (w3 + gap) + w3 / 2
    arrow(ax, (xt, 0.42), (xt, 0.35), "grouped by mechanism", tx=-0.012, ha="right")
    box(ax, 0.0, 0.0, 0.54, 0.35, "Pre-1900 control groups",
        f"original: {n_ctrl} groups of test formulas\n"
        f"regenerated: the same {n_regen} groups resampled\nwith the later families' sampling",
        edge=S.CLASSICAL)

    box(ax, 0.62, 0.70, 0.38, 0.30, "Later law families",
        f"{n_all} families after 1900\n({n_early} from 1901–1950,\n{n_late} after 1950)", edge=S.LATER)
    box(ax, 0.62, 0.36, 0.38, 0.28, "Same-generator set",
        "later formulas produced by\nthe generator of the training\ncorpus (1901–1950)", edge=S.LATER)
    box(ax, 0.62, 0.0, 0.38, 0.30, "Measured data",
        f"historical datasets and\n{measured['datasets']} NIST series in\n{measured['mechanism_groups']} mechanism groups",
        edge=S.INK)
    arrow(ax, (0.54, 0.25), (0.62, 0.80))
    arrow(ax, (0.54, 0.20), (0.62, 0.50))
    S.save(fig, OUT)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
