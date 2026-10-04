"""Publication figure for the frozen large-scale learned EPOCH clock audit.

Inputs are the two fail-closed evaluation JSON files and their paired clock
comparison.  This script is visualization-only: it does not fit a detector,
select a score, or change any preregistered result.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PHASE_A = RESULTS / "phase_a_pre1900_v3_evaluation.json"
PHASE_B = RESULTS / "phase_b_pre1950_v3_evaluation.json"
CLOCK = RESULTS / "epoch_clock_v3_comparison.json"
RANDOM_PHASE_A = RESULTS / "random_phase_a_pre1900_v3_evaluation.json"
RANDOM_PHASE_B = RESULTS / "random_phase_b_pre1950_v3_evaluation.json"
RANDOM_CLOCK = RESULTS / "random_epoch_clock_v3_comparison.json"
MATCHED_PHASE_A = RESULTS / "matched_pre1900_domain_control_trained_v3.json"
MATCHED_RANDOM_PHASE_A = RESULTS / "matched_pre1900_domain_control_random_v3.json"
OUT_PNG = ROOT / "figures" / "figure_learned_epoch_clock_v3.png"
OUT_PDF = ROOT / "figures" / "figure_learned_epoch_clock_v3.pdf"

BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
PURPLE = "#8C6BB1"
GREY = "#7F8C8D"
LIGHT_GREY = "#D9D9D6"
INK = "#252525"
BG = "#FBFBF9"
METRIC = "one_class_ensemble"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(payload: dict, set_name: str, metric: str = METRIC) -> np.ndarray:
    rows = payload["sets"][set_name]["scores"]
    result = np.asarray([row[metric] for row in rows.values()], dtype=float)
    return result[np.isfinite(result)]


def roc_panel(ax: plt.Axes, old: dict, new: dict, matched: dict) -> None:
    curves = [
        (
            values(old, "known_internal"),
            values(old, "curated_all_post1900"),
            "1899: native controls → all later", ORANGE,
            old["comparisons"]["known_internal_vs_curated_all_post1900"],
        ),
        (
            values(matched, "generator_matched_pre1900_controls"),
            values(old, "curated_all_post1900"),
            "1899: matched controls → all later", GREEN,
            matched["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
        ),
        (
            values(new, "known_internal"),
            values(new, "curated_post1950"),
            "1950: known controls → post-1950", BLUE,
            new["comparisons"]["known_internal_vs_curated_post1950"],
        ),
    ]
    for negative, positive, label, color, result in curves:
        y = np.r_[np.zeros(len(negative), int), np.ones(len(positive), int)]
        score = np.r_[negative, positive]
        fpr, tpr, _ = roc_curve(y, score)
        lo, hi = result["ci95"]
        ax.step(
            fpr, tpr, where="post", linewidth=2.2, color=color,
            label=f"{label}: {result['auroc']:.3f} [{lo:.3f}, {hi:.3f}]",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, color=GREY)
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.015),
           xlabel="false-positive rate across known-horizon groups",
           ylabel="true-positive rate across later families")
    ax.set_title("a   Frozen temporal discrimination", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=7.7)


def paired_panel(ax: plt.Axes, clock: dict, random_clock: dict | None) -> None:
    paired = clock["paired_scores"]
    old = np.asarray([row["pre1900_score"] for row in paired.values()], float)
    new = np.asarray([row["pre1950_score"] for row in paired.values()], float)
    decreases = old > new
    for left, right, decreased in zip(old, new, decreases):
        ax.plot(
            [0, 1], [left, right],
            color=ORANGE if decreased else LIGHT_GREY,
            alpha=0.34 if decreased else 0.65, linewidth=0.8, zorder=1,
        )
    ax.scatter(np.zeros(len(old)), old, s=17, color=ORANGE, alpha=0.72,
               linewidth=0, zorder=2)
    ax.scatter(np.ones(len(new)), new, s=17, color=BLUE, alpha=0.72,
               linewidth=0, zorder=2)
    medians = [np.median(old), np.median(new)]
    ax.plot([0, 1], medians, color=INK, linewidth=2.1, zorder=3)
    ax.scatter([0, 1], medians, marker="D", s=42, color=INK, zorder=4)
    lo, hi = clock["median_collapse_ci95"]
    fraction = clock["fraction_decreased_among_non_ties"]
    random_line = ""
    if random_clock is not None:
        rlo, rhi = random_clock["median_collapse_ci95"]
        random_line = (
            f"\nuntrained control: {random_clock['median_collapse']:.3f} "
            f"[{rlo:.3f}, {rhi:.3f}]"
        )
    ax.text(
        0.03, 0.03,
        f"median collapse {clock['median_collapse']:.3f} "
        f"[{lo:.3f}, {hi:.3f}]\n"
        f"decreased: {fraction:.0%}; sign-test p={clock['exact_two_sided_sign_test_p']:.3g}"
        f"{random_line}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8,
        bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
    )
    ax.set_xticks([0, 1], ["1899 horizon", "1950 horizon"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("calibrated one-class anomaly score")
    ax.set_title(
        f"b   Same {len(paired)} families after clock advance", loc="left",
        fontsize=10, fontweight="bold",
    )


def summary_panel(
    ax: plt.Axes,
    old: dict,
    new: dict,
    random_old: dict,
    random_new: dict,
    matched: dict,
    matched_random: dict,
) -> None:
    entries = [
        (
            old["comparisons"]["pre1900_internal_vs_generated_1901_1950"],
            random_old["comparisons"]["pre1900_internal_vs_generated_1901_1950"],
            "1899 native →\ngenerator-gated",
        ),
        (
            old["comparisons"]["known_internal_vs_curated_all_post1900"],
            random_old["comparisons"]["known_internal_vs_curated_all_post1900"],
            "1899 native →\nall later",
        ),
        (
            matched["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
            matched_random["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
            "1899 matched →\nall later*",
        ),
        (
            new["comparisons"]["known_internal_vs_curated_post1950"],
            random_new["comparisons"]["known_internal_vs_curated_post1950"],
            "1950 known →\npost-1950",
        ),
    ]
    estimates = np.asarray([entry[0]["auroc"] for entry in entries])
    cis = np.asarray([entry[0]["ci95"] for entry in entries])
    x = np.arange(len(entries))
    colors = [GREY, ORANGE, GREEN, BLUE]
    for position, estimate, interval, color in zip(x, estimates, cis, colors):
        ax.errorbar(
            position, estimate,
            yerr=np.asarray([[estimate - interval[0]], [interval[1] - estimate]]),
            fmt="none", ecolor=color, elinewidth=1.3, capsize=3, zorder=2,
        )
    ax.scatter(x, estimates, s=46, c=colors, zorder=3)
    random_estimates = np.asarray([entry[1]["auroc"] for entry in entries])
    random_cis = np.asarray([entry[1]["ci95"] for entry in entries])
    ax.errorbar(
        x + 0.12, random_estimates,
        yerr=np.vstack((
            random_estimates - random_cis[:, 0],
            random_cis[:, 1] - random_estimates,
        )),
        fmt="o", markersize=4.4, markerfacecolor=BG,
        markeredgecolor=GREY, color=GREY, ecolor=GREY,
        elinewidth=0.9, capsize=2, zorder=2,
        label="same-seed, zero-optimization control",
    )
    for position, estimate in zip(x, estimates):
        ax.text(position, min(1.04, estimate + 0.055), f"{estimate:.3f}",
                ha="center", va="bottom", fontsize=7.4)
    ax.axhline(0.5, linestyle="--", linewidth=0.9, color=GREY)
    ax.set_xticks(x, [entry[2] for entry in entries], fontsize=7.4)
    ax.set_ylim(0.18, 1.08)
    ax.set_ylabel("family/group AUROC (95% grouped-bootstrap CI)")
    ax.set_title("c   Frozen comparisons plus matched-domain diagnostic", loc="left",
                 fontsize=10, fontweight="bold")
    ax.text(
        0.99, 0.02, "* post-score generator-matched diagnostic",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7, color=GREY,
    )
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")


def component_panel(ax: plt.Axes, old: dict, new: dict) -> None:
    components = [
        ("graph_formula_knn_distance", "graph-formula\nkNN"),
        ("graph_identity_knn_distance", "identity-centroid\nkNN"),
        ("point_formula_knn_distance", "point-prototype\nkNN"),
        ("point_shrinkage_mahalanobis", "point-prototype\nMahalanobis"),
        (METRIC, "fixed four-score\nensemble"),
    ]
    old_auc = old["component_aurocs_descriptive"][
        "known_internal_vs_curated_all_post1900"
    ]
    new_auc = new["component_aurocs_descriptive"][
        "known_internal_vs_curated_post1950"
    ]
    x = np.arange(len(components))
    width = 0.35
    ax.bar(x - width / 2, [old_auc[key] for key, _ in components],
           width, color=ORANGE, label="1899 horizon → all post-1900")
    ax.bar(x + width / 2, [new_auc[key] for key, _ in components],
           width, color=BLUE, label="1950 horizon → post-1950")
    ax.axhline(0.5, linestyle="--", linewidth=0.9, color=GREY)
    ax.set_xticks(x, [label for _, label in components], fontsize=7.4)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC (descriptive component audit)")
    ax.set_title("d   What contributes to the frozen score", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color="#E7E7E3", linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def main() -> None:
    old = load(PHASE_A)
    new = load(PHASE_B)
    clock = load(CLOCK)
    random_old = load(RANDOM_PHASE_A) if RANDOM_PHASE_A.exists() else None
    random_new = load(RANDOM_PHASE_B) if RANDOM_PHASE_B.exists() else None
    random_clock = load(RANDOM_CLOCK) if RANDOM_CLOCK.exists() else None
    matched = load(MATCHED_PHASE_A)
    matched_random = load(MATCHED_RANDOM_PHASE_A)
    assert old["knowledge_cutoff"] == 1899
    assert new["knowledge_cutoff"] == 1950
    assert old["primary_metric"] == new["primary_metric"] == METRIC
    assert clock["score"] == METRIC

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), facecolor=BG)
    if random_old is None or random_new is None or random_clock is None:
        raise FileNotFoundError("the complete same-seed random-control suite is required")
    roc_panel(axes[0, 0], old, new, matched)
    paired_panel(axes[0, 1], clock, random_clock)
    summary_panel(
        axes[1, 0], old, new, random_old, random_new, matched, matched_random
    )
    component_panel(axes[1, 1], old, new)
    for ax in axes.flat:
        style(ax)

    fig.suptitle(
        "Large-scale classical-only learning: strong 1899 ranking, failed post-1950 transfer",
        x=0.06, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.951,
        "Point-cloud queries only; no anomaly templates or threshold selection; "
        "the 1950-horizon failure and post-score matched-generator diagnostic are shown.",
        ha="left", va="top", fontsize=8.5, color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.09,
                        hspace=0.37, wspace=0.25)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=BG)
    print(json.dumps({"png": str(OUT_PNG), "pdf": str(OUT_PDF)}, indent=2))


if __name__ == "__main__":
    main()
