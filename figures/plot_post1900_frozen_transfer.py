"""Plot rank-based frozen transfer to post-1900 formula families.

The main figure deliberately avoids selecting an alert threshold.  Its estimand
is family-level discrimination (AUROC) between post-1900 formula families and
cited pre-1900 families held out from encoder training.  Fixed-threshold alert
counts remain available in the machine-readable audit and Supplement S15.
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
RESULT = ROOT / "results" / "frozen_post1900_formula_evaluation.json"
OUT_PNG = ROOT / "figures" / "figure_post1900_frozen_transfer.png"
OUT_PDF = ROOT / "figures" / "figure_post1900_frozen_transfer.pdf"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#7F8C8D"
INK = "#252525"
SHAPES = [
    ("declared_dictionary_overlap", "four-form\noverlap"),
    ("smooth_composite", "smooth\ncomposite"),
    ("threshold_or_piecewise", "threshold /\npiecewise"),
    ("special_function_or_integral", "integral /\nspecial"),
    ("nonmonotone_or_oscillatory", "nonmonotone /\noscillatory"),
]


def arrays(payload: dict, score_key: str) -> tuple[np.ndarray, np.ndarray]:
    controls = np.asarray(
        [row[score_key] for row in payload["heldout_pre1900_controls"]], float
    )
    later = np.asarray([row[score_key] for row in payload["post1900_families"]], float)
    return controls[np.isfinite(controls)], later[np.isfinite(later)]


def pooled_percentile(controls: np.ndarray, later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mid-ranks on a common 0--1 scale; AUROC is rank invariant."""
    values = np.r_[controls, later]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average tied ranks without adding a scipy dependency.
    for value in np.unique(values):
        at = np.flatnonzero(values == value)
        ranks[at] = np.mean(ranks[at])
    ranks /= max(1, len(values) - 1)
    return ranks[: len(controls)], ranks[len(controls) :]


def roc_panel(ax, payload: dict) -> None:
    primary = payload["primary_family_level"]
    for key, label, color, result_key in (
        ("learned_score", "learned component", BLUE, "learned"),
        ("stat_score", "four-form component", ORANGE, "statistical_four_form"),
    ):
        controls, later = arrays(payload, key)
        labels = np.r_[np.zeros(len(controls), int), np.ones(len(later), int)]
        scores = np.r_[controls, later]
        fpr, tpr, _ = roc_curve(labels, scores)
        result = primary[result_key]
        auc = result["auroc_post1900_vs_heldout_pre1900"]
        lo, hi = result["family_bootstrap_95_ci"]
        ax.step(
            fpr, tpr, where="post", color=color, linewidth=2.1,
            label=f"{label}: AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]",
        )
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=0.9)
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.015),
           xlabel="false-positive rate across held-out pre-1900 families",
           ylabel="true-positive rate across post-1900 families")
    ax.set_title("a   Frozen family-level discrimination", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(color="#E5E5E5", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def distribution_panel(ax, payload: dict) -> None:
    rng = np.random.default_rng(13)
    entries = []
    for centre, key, color in ((0, "learned_score", BLUE), (3, "stat_score", ORANGE)):
        controls, later = arrays(payload, key)
        control_ranks, later_ranks = pooled_percentile(controls, later)
        entries.extend([
            (centre, control_ranks, "pre-1900\nholdouts", "white", color),
            (centre + 1, later_ranks, "post-1900\nformulas", color, color),
        ])
    for x, values, _, face, edge in entries:
        bp = ax.boxplot(
            [values], positions=[x], widths=0.55, patch_artist=True,
            showfliers=False, medianprops={"color": INK, "linewidth": 1.3},
            boxprops={"facecolor": face, "edgecolor": edge, "linewidth": 1.1},
            whiskerprops={"color": edge, "linewidth": 1.0},
            capprops={"color": edge, "linewidth": 1.0},
        )
        del bp
        jitter = rng.normal(0, 0.055, len(values))
        ax.scatter(
            x + jitter, values, s=18, facecolor=face, edgecolor=edge,
            linewidth=0.65, alpha=0.72, zorder=3,
        )
    ax.set_xticks([0, 1, 3, 4], [entry[2] for entry in entries])
    ax.text(0.5, 1.075, "learned", ha="center", fontsize=8.5, color=BLUE)
    ax.text(3.5, 1.075, "four-form", ha="center", fontsize=8.5, color=ORANGE)
    ax.set_ylim(-0.04, 1.13)
    ax.set_ylabel("pooled family-score percentile")
    ax.set_title("b   Continuous scores, without an alert cut-off", loc="left",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def auc_against_controls(controls: np.ndarray, positives: np.ndarray) -> float:
    """Mann--Whitney form of AUROC, with half credit for ties."""
    comparisons = positives[:, None] - controls[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def structure_panel(ax, payload: dict) -> None:
    rows = payload["post1900_families"]
    learned_controls, _ = arrays(payload, "learned_score")
    stat_controls, _ = arrays(payload, "stat_score")
    x = np.arange(len(SHAPES))
    width = 0.34
    for offset, key, controls, color, label in (
        (-width / 2, "learned_score", learned_controls, BLUE, "learned"),
        (width / 2, "stat_score", stat_controls, ORANGE, "four-form"),
    ):
        values = []
        for shape, _ in SHAPES:
            positive = np.asarray(
                [row[key] for row in rows if row["shape_class"] == shape], float
            )
            positive = positive[np.isfinite(positive)]
            values.append(auc_against_controls(controls, positive))
        ax.bar(x + offset, values, width=width * 0.92, color=color, alpha=0.9,
               label=label)
    counts = [sum(row["shape_class"] == shape for row in rows) for shape, _ in SHAPES]
    for position, count in zip(x, counts):
        ax.text(position, 1.025, f"n={count}", ha="center", va="bottom",
                fontsize=7, color="#555555")
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, [label for _, label in SHAPES])
    ax.set_ylim(0, 1.11)
    ax.set_ylabel("AUROC versus the same pre-1900 holdouts")
    ax.set_title("c   Discrimination varies with functional structure", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=8)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(12.2, 7.7), facecolor="#FBFBF9")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_facecolor("#FBFBF9")

    roc_panel(ax_a, payload)
    distribution_panel(ax_b, payload)
    structure_panel(ax_c, payload)

    fig.suptitle(
        "Frozen transfer to 55 post-1900 formula families",
        x=0.055, y=0.985, ha="left", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.055, 0.949,
        "Family-level ranking against 11 cited pre-1900 holdouts; "
        "440 generated point clouds; no retraining or recalibration.",
        ha="left", va="top", fontsize=8.5, color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.875, bottom=0.105,
                        hspace=0.38, wspace=0.24)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(json.dumps({"png": str(OUT_PNG), "pdf": str(OUT_PDF)}, indent=2))


if __name__ == "__main__":
    main()
