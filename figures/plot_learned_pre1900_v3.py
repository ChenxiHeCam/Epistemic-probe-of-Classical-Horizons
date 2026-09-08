"""Publication figure for the pre-1900-only EPOCH learned-v3 audit.

The figure uses only the frozen 1899-horizon checkpoint, its same-seed
zero-optimization control and the explicitly post-score generator-matched
negative-control diagnostic.  It performs visualization only.
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
RANDOM_PHASE_A = RESULTS / "random_phase_a_pre1900_v3_evaluation.json"
MATCHED_PHASE_A = RESULTS / "matched_pre1900_domain_control_trained_v3.json"
MATCHED_RANDOM_PHASE_A = RESULTS / "matched_pre1900_domain_control_random_v3.json"
OUT_PNG = ROOT / "figures" / "figure_learned_pre1900_v3.png"
OUT_PDF = ROOT / "figures" / "figure_learned_pre1900_v3.pdf"

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


def add_roc(
    ax: plt.Axes,
    negative: np.ndarray,
    positive: np.ndarray,
    result: dict,
    label: str,
    color: str,
) -> None:
    labels = np.r_[np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)]
    scores = np.r_[negative, positive]
    fpr, tpr, _ = roc_curve(labels, scores)
    lo, hi = result["ci95"]
    ax.step(
        fpr,
        tpr,
        where="post",
        linewidth=2.2,
        color=color,
        label=f"{label}: {result['auroc']:.3f} [{lo:.3f}, {hi:.3f}]",
    )


def roc_panel(ax: plt.Axes, phase_a: dict, matched: dict) -> None:
    native = values(phase_a, "known_internal")
    near = values(phase_a, "curated_1901_1950")
    all_later = values(phase_a, "curated_all_post1900")
    matched_controls = values(matched, "generator_matched_pre1900_controls")
    add_roc(
        ax,
        native,
        near,
        phase_a["comparisons"]["known_internal_vs_curated_1901_1950"],
        "native controls vs 1901-1950",
        ORANGE,
    )
    add_roc(
        ax,
        native,
        all_later,
        phase_a["comparisons"]["known_internal_vs_curated_all_post1900"],
        "native controls vs all 55 later",
        BLUE,
    )
    add_roc(
        ax,
        matched_controls,
        all_later,
        matched["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
        "matched controls vs all 55 later*",
        GREEN,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, color=GREY)
    ax.set(
        xlim=(-0.015, 1.015),
        ylim=(-0.015, 1.015),
        xlabel="false-positive rate across pre-1900 control groups",
        ylabel="true-positive rate across later formula families",
    )
    ax.set_title("a   Frozen 1899-horizon ranking", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=7.4)


def distribution_panel(ax: plt.Axes, phase_a: dict, matched: dict) -> None:
    groups = [
        values(phase_a, "known_internal"),
        values(matched, "generator_matched_pre1900_controls"),
        values(phase_a, "curated_1901_1950"),
        values(phase_a, "curated_post1950"),
    ]
    labels = ["native\npre-1900", "matched\npre-1900*", "1901-1950", "later than\n1950"]
    colors = [GREY, GREEN, ORANGE, PURPLE]
    bp = ax.boxplot(
        groups,
        positions=np.arange(4),
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": GREY, "linewidth": 0.9},
        capprops={"color": GREY, "linewidth": 0.9},
        boxprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.23)
    rng = np.random.default_rng(20260908)
    for index, (group, color) in enumerate(zip(groups, colors)):
        jitter = rng.uniform(-0.15, 0.15, len(group))
        ax.scatter(index + jitter, group, s=18, color=color, alpha=0.78, linewidth=0, zorder=3)
        ax.text(
            index,
            1.015,
            f"n={len(group)}",
            ha="center",
            va="top",
            fontsize=7.3,
            color=GREY,
        )
    ax.set_xticks(np.arange(4), labels)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.02, 1.04)
    ax.set_ylabel("calibrated one-class anomaly score")
    ax.set_title("b   Frozen family/group score distributions", loc="left", fontsize=10,
                 fontweight="bold")
    ax.text(
        0.99,
        0.02,
        "* post-score generator-matched controls",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=GREY,
    )


def comparison_panel(
    ax: plt.Axes,
    phase_a: dict,
    random_a: dict,
    matched: dict,
    matched_random: dict,
) -> None:
    entries = [
        (
            phase_a["comparisons"]["pre1900_internal_vs_generated_1901_1950"],
            random_a["comparisons"]["pre1900_internal_vs_generated_1901_1950"],
            "native generator-\ngated 1901-1950",
        ),
        (
            phase_a["comparisons"]["known_internal_vs_curated_1901_1950"],
            random_a["comparisons"]["known_internal_vs_curated_1901_1950"],
            "native controls\nvs 1901-1950",
        ),
        (
            phase_a["comparisons"]["known_internal_vs_curated_all_post1900"],
            random_a["comparisons"]["known_internal_vs_curated_all_post1900"],
            "native controls\nvs all later",
        ),
        (
            matched["comparisons"]["matched_pre1900_vs_curated_1901_1950"],
            matched_random["comparisons"]["matched_pre1900_vs_curated_1901_1950"],
            "matched controls\nvs 1901-1950*",
        ),
        (
            matched["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
            matched_random["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
            "matched controls\nvs all later*",
        ),
    ]
    x = np.arange(len(entries), dtype=float)
    for position, (trained, random, _) in enumerate(entries):
        for offset, result, color, filled in (
            (-0.08, trained, ORANGE, True),
            (0.08, random, GREY, False),
        ):
            estimate = result["auroc"]
            lo, hi = result["ci95"]
            ax.errorbar(
                position + offset,
                estimate,
                yerr=np.asarray([[estimate - lo], [hi - estimate]]),
                fmt="o",
                markersize=5.0,
                markerfacecolor=color if filled else BG,
                markeredgecolor=color,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )
        ax.text(position - 0.08, min(1.055, trained["auroc"] + 0.05),
                f"{trained['auroc']:.3f}", ha="center", va="bottom", fontsize=7.0)
    ax.axhline(0.5, linestyle="--", linewidth=0.9, color=GREY)
    ax.set_xticks(x, [entry[2] for entry in entries], fontsize=7.1)
    ax.set_ylim(0.28, 1.08)
    ax.set_ylabel("family/group AUROC (grouped-bootstrap 95% CI)")
    ax.set_title("c   Trained representation versus zero optimization", loc="left",
                 fontsize=10, fontweight="bold")
    ax.scatter([], [], s=28, color=ORANGE, label="pre-1900 trained")
    ax.scatter([], [], s=28, facecolor=BG, edgecolor=GREY,
               label="same-seed zero optimization")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    ax.text(0.99, 0.02, "* post-score diagnostic", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color=GREY)


def component_panel(ax: plt.Axes, phase_a: dict, random_a: dict) -> None:
    components = [
        ("graph_formula_knn_distance", "graph-formula\nkNN"),
        ("graph_identity_knn_distance", "identity-centroid\nkNN"),
        ("point_formula_knn_distance", "point-prototype\nkNN"),
        ("point_shrinkage_mahalanobis", "point-prototype\nMahalanobis"),
        (METRIC, "fixed four-score\nensemble"),
    ]
    key = "known_internal_vs_curated_all_post1900"
    trained = phase_a["component_aurocs_descriptive"][key]
    random = random_a["component_aurocs_descriptive"][key]
    x = np.arange(len(components))
    width = 0.35
    ax.bar(x - width / 2, [trained[name] for name, _ in components], width,
           color=ORANGE, label="pre-1900 trained")
    ax.bar(x + width / 2, [random[name] for name, _ in components], width,
           color=LIGHT_GREY, edgecolor=GREY, linewidth=0.8,
           label="zero optimization")
    ax.axhline(0.5, linestyle="--", linewidth=0.9, color=GREY)
    ax.set_xticks(x, [label for _, label in components], fontsize=7.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC against all 55 later families")
    ax.set_title("d   Contribution of the frozen score components", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.4, loc="lower left")


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color="#E7E7E3", linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def main() -> None:
    phase_a = load(PHASE_A)
    random_a = load(RANDOM_PHASE_A)
    matched = load(MATCHED_PHASE_A)
    matched_random = load(MATCHED_RANDOM_PHASE_A)
    assert phase_a["knowledge_cutoff"] == random_a["knowledge_cutoff"] == 1899
    assert phase_a["primary_metric"] == random_a["primary_metric"] == METRIC

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), facecolor=BG)
    roc_panel(axes[0, 0], phase_a, matched)
    distribution_panel(axes[0, 1], phase_a, matched)
    comparison_panel(axes[1, 0], phase_a, random_a, matched, matched_random)
    component_panel(axes[1, 1], phase_a, random_a)
    for ax in axes.flat:
        style(ax)

    fig.suptitle(
        "Classical-only representation learning at the 1899 knowledge horizon",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.951,
        "One frozen pre-1900 model; later families are evaluation only; no anomaly templates, "
        "threshold selection, units, dates or source labels reach the query encoder.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.09,
                        hspace=0.37, wspace=0.25)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=BG)
    print(json.dumps({"png": str(OUT_PNG), "pdf": str(OUT_PDF)}, indent=2))


if __name__ == "__main__":
    main()
