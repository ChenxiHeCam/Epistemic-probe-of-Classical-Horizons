"""Figure: large encoder at the 1899 horizon (manuscript Fig. 3).

Visualization only: every number is read from the evaluation JSON files.
Terms follow the manuscript: original and regenerated pre-1900 control groups,
the same-generator set, the untrained network, and the four distance components.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = ROOT / "results"
PHASE_A = RESULTS / "phase_a_pre1900_v3_evaluation.json"
RANDOM_PHASE_A = RESULTS / "random_phase_a_pre1900_v3_evaluation.json"
MATCHED_PHASE_A = RESULTS / "matched_pre1900_domain_control_trained_v3.json"
MATCHED_RANDOM_PHASE_A = RESULTS / "matched_pre1900_domain_control_random_v3.json"
OUT = ROOT / "figures" / "figure_large_encoder"
METRIC = "one_class_ensemble"
NL = chr(10)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(payload: dict, set_name: str, metric: str = METRIC) -> np.ndarray:
    rows = payload["sets"][set_name]["scores"]
    result = np.asarray([row[metric] for row in rows.values()], dtype=float)
    return result[np.isfinite(result)]


def fmt(result: dict) -> str:
    lo, hi = result["ci95"]
    return f"{result['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"


def roc_panel(ax, phase_a: dict, matched: dict) -> None:
    original = values(phase_a, "known_internal")
    regenerated = values(matched, "generator_matched_pre1900_controls")
    near = values(phase_a, "curated_1901_1950")
    later = values(phase_a, "curated_all_post1900")
    rows = [
        (original, later, phase_a["comparisons"]["known_internal_vs_curated_all_post1900"],
         "55 later families, original controls", "-"),
        (original, near, phase_a["comparisons"]["known_internal_vs_curated_1901_1950"],
         "38 families 1901–1950, original controls", "--"),
        (regenerated, later, matched["comparisons"]["matched_pre1900_vs_curated_all_post1900"],
         "55 later families, regenerated controls", ":"),
    ]
    for neg, pos, result, label, ls in rows:
        y = np.r_[np.zeros(len(neg), int), np.ones(len(pos), int)]
        fpr, tpr, _ = roc_curve(y, np.r_[neg, pos])
        ax.step(fpr, tpr, where="post", color=S.LEARNED, linestyle=ls, linewidth=1.3,
                label=f"{label}\nAUROC {fmt(result)}")
    ax.plot([0, 1], [0, 1], color=S.LIGHT, linewidth=0.8, zorder=0)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.03), xlabel="False-positive rate (pre-1900 control groups)",
           ylabel="True-positive rate (later law families)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=6.5, handlelength=2.4, labelspacing=0.55,
              frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    S.panel(ax, "a", "Ranking of later law families")


def distribution_panel(ax, phase_a: dict, matched: dict) -> None:
    groups = [
        (values(phase_a, "known_internal"), "Pre-1900\noriginal", S.CLASSICAL, "o", True),
        (values(matched, "generator_matched_pre1900_controls"), "Pre-1900\nregenerated", S.CLASSICAL, "o", False),
        (values(phase_a, "curated_1901_1950"), "1901–\n1950", S.LATER, "s", True),
        (values(phase_a, "curated_post1950"), "After\n1950", S.LATER, "s", False),
    ]
    rng = np.random.default_rng(20260908)
    for i, (g, _, color, marker, filled) in enumerate(groups):
        ax.boxplot([g], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   medianprops={"color": S.INK, "linewidth": 1.0},
                   whiskerprops={"color": S.MUTED, "linewidth": 0.6},
                   capprops={"color": S.MUTED, "linewidth": 0.6},
                   boxprops={"facecolor": "none", "edgecolor": S.MUTED, "linewidth": 0.6})
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(g)), g, s=9, marker=marker, linewidths=0.6,
                   facecolors=color if filled else "white", edgecolors=color, alpha=0.9, zorder=3)
        ax.text(i, 1.03, f"n = {len(g)}", ha="center", va="bottom", fontsize=6.5, color=S.MUTED)
    ax.set_xticks(range(4), [g[1] for g in groups])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.03, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Learned score, large encoder\n(family or group median)")
    S.grid(ax)
    S.panel(ax, "b", "Learned scores by group")


def comparison_panel(ax, phase_a: dict, random_a: dict, matched: dict, matched_random: dict) -> None:
    entries = [
        ("pre1900_internal_vs_generated_1901_1950", phase_a, random_a, "Same-generator\n1901–1950"),
        ("known_internal_vs_curated_1901_1950", phase_a, random_a, "Original\n1901–1950"),
        ("known_internal_vs_curated_all_post1900", phase_a, random_a, "Original\nall 55"),
        ("matched_pre1900_vs_curated_1901_1950", matched, matched_random, "Regenerated\n1901–1950"),
        ("matched_pre1900_vs_curated_all_post1900", matched, matched_random, "Regenerated\nall 55"),
    ]
    for i, (key, trained_src, random_src, _) in enumerate(entries):
        for dx, res, color, filled in ((-0.11, trained_src["comparisons"][key], S.LEARNED, True),
                                       (0.11, random_src["comparisons"][key], S.UNTRAINED, False)):
            est = res["auroc"]
            lo, hi = res["ci95"]
            ax.errorbar(i + dx, est, yerr=[[est - lo], [hi - est]], fmt="o", ms=3.6,
                        mfc=color if filled else "white", mec=color, ecolor=color,
                        elinewidth=0.8, capsize=1.8, mew=0.8, zorder=3)
        t = trained_src["comparisons"][key]
        if i == 0:  # beside the point, clear of the untrained network's interval
            ax.text(i - 0.2, t["auroc"], f"{t['auroc']:.3f}", ha="right", va="center",
                    fontsize=6.5, color=S.LEARNED)
        else:
            ax.text(i - 0.11, t["ci95"][1] + 0.015, f"{t['auroc']:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=S.LEARNED)
    ax.axvline(0.5, ymax=0.84, color=S.LIGHT, linewidth=0.8)
    ax.axhline(0.5, color=S.MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xticks(range(len(entries)), [e[3].replace(NL, " ") for e in entries], fontsize=6.5,
                  rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.95, len(entries) - 0.5)
    ax.set_ylim(0.25, 1.17)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    ax.set_ylabel("AUROC (95% bootstrap interval)")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=3.6, mfc=S.LEARNED, mec=S.LEARNED, label="Large encoder"),
        Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=S.UNTRAINED, label="Untrained network"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=6.5, ncol=2)
    S.grid(ax)
    S.panel(ax, "c", "Trained and untrained network by control set")


def component_panel(ax, phase_a: dict, random_a: dict) -> None:
    components = [
        ("graph_formula_knn_distance", "Nearest formula embedding"),
        ("graph_identity_knn_distance", "Law centroid"),
        ("point_formula_knn_distance", "Point-cloud prototype"),
        ("point_shrinkage_mahalanobis", "Mahalanobis distance"),
        (METRIC, "Equal-weight mean"),
    ]
    key = "known_internal_vs_curated_all_post1900"
    trained = phase_a["component_aurocs_descriptive"][key]
    random = random_a["component_aurocs_descriptive"][key]
    x = np.arange(len(components))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, [trained[k] for k, _ in components], w, color=S.LEARNED, label="Large encoder")
    ax.bar(x + w / 2 + 0.01, [random[k] for k, _ in components], w, color="white", edgecolor=S.UNTRAINED,
           linewidth=0.8, label="Untrained network")
    for xi, (k, _) in zip(x, components):
        ax.text(xi - w / 2 - 0.01, trained[k] + 0.015, f"{trained[k]:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=S.LEARNED)
    ax.axhline(0.5, color=S.MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xticks(x, [c[1] for c in components], fontsize=6.5,
                  rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("AUROC (55 later vs nine groups)")
    ax.legend(loc="upper center", fontsize=6.5, ncol=2, bbox_to_anchor=(0.5, 1.03))
    S.grid(ax)
    S.panel(ax, "d", "The four distance components")


def main() -> None:
    S.apply()
    phase_a, random_a = load(PHASE_A), load(RANDOM_PHASE_A)
    matched, matched_random = load(MATCHED_PHASE_A), load(MATCHED_RANDOM_PHASE_A)
    assert phase_a["knowledge_cutoff"] == random_a["knowledge_cutoff"] == 1899
    assert phase_a["primary_metric"] == random_a["primary_metric"] == METRIC

    fig = plt.figure(figsize=(S.WIDTH, 142 * S.MM))
    gs = fig.add_gridspec(2, 2, left=0.15, right=0.99, top=0.95, bottom=0.16, hspace=0.62, wspace=0.3)
    roc_panel(fig.add_subplot(gs[0, 0]), phase_a, matched)
    distribution_panel(fig.add_subplot(gs[0, 1]), phase_a, matched)
    comparison_panel(fig.add_subplot(gs[1, 0]), phase_a, random_a, matched, matched_random)
    component_panel(fig.add_subplot(gs[1, 1]), phase_a, random_a)
    S.save(fig, OUT)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
