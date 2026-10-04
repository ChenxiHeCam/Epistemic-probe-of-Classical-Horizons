"""Panels for the paired prediction tests, observation effects, boundary calibration and the compact
encoder, and the observation-and-calibration figure (manuscript Fig. 5).

The p-value panel is drawn into the historical-case figure (make_fig_cases.py) and the compact-encoder
panel into the screening figure (make_fig_screening.py).

Visualization only: every number is read from the results JSON files. Colours follow
figure_scripts/epoch_style.py: pre-1900 predictions blue, successor predictions vermillion,
the extrapolation screen green, the compact encoder purple, controls grey.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "figures"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def pvalue_panel(ax, letter: str = "a", legend: str = "right") -> None:
    """The same measurements under the pre-1900 prediction, its successor and a wrong-mechanism form."""
    clock = load("theory_clock_experiment.json")
    physical_controls = load("physical_irrelevant_horizon_controls.json")
    names = {"specific_heat_Cu": "Copper heat\ncapacity", "FIRAS": "FIRAS", "Bertozzi": "Bertozzi",
             "Onnes": "Onnes", "Millikan": "Millikan"}
    labels = [names.get(r["case"], r["case"]) for r in clock["cases"]]
    old_p = np.array([r["old_horizon"]["pvalue"] for r in clock["cases"]])
    new_p = np.array([r["advanced_horizon"]["pvalue"] for r in clock["cases"]])
    wrong_p = np.array([r["pvalue"] for r in physical_controls["cases"]])
    y = np.arange(len(labels))
    for yi, left, right in zip(y, old_p, new_p):
        ax.plot([left, right], [yi, yi], color=S.LIGHT, lw=1.2, zorder=1)
    ax.scatter(old_p, y - 0.26, color=S.CLASSICAL, s=18, zorder=3, label="Pre-1900 prediction")
    ax.scatter(new_p, y, marker="s", facecolors="white", edgecolors=S.LATER, linewidths=1.1, s=18, zorder=3,
               label="Successor prediction")
    ax.scatter(wrong_p, y + 0.26, color=S.UNTRAINED, marker="x", linewidths=0.9, s=16, zorder=3,
               label="Wrong-mechanism form")
    ax.axvline(0.05, color=S.INK, ls=(0, (3, 2)), lw=0.7)
    ax.text(0.05, -0.85, "  p = 0.05", fontsize=6.5, color=S.MUTED, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 1.6)
    ax.set_ylim(len(labels) - 0.5, -1.2)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Goodness-of-fit p-value")
    if legend == "right":
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5, handletextpad=0.3,
                  borderaxespad=0.3, labelspacing=0.8)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.3), ncol=2, fontsize=6.5, columnspacing=1.0,
                  handletextpad=0.3)
    S.grid(ax, "x")
    S.panel(ax, letter, "Pre-1900 and successor predictions", x=-0.2)


def instrument_panel(ax, letter: str = "a") -> None:
    """False-positive rate of the extrapolation screen under simulated instrument effects."""
    robust = load("instrument_robustness.json")
    kinds = ["calibration_drift", "sensor_saturation", "censoring", "x_axis_error", "outliers",
             "correlated_noise", "heteroscedasticity"]
    names = ["Drift", "Saturation", "Censoring", "Abscissa error", "Outliers", "Correlated noise", "Heteroscedastic"]
    fprs = []
    for kind in kinds:
        row = max((r for r in robust["summary"] if r["corruption"] == kind), key=lambda r: r["severity"])
        fprs.append(row["fpr_at_nominal_5pct"])
    xb = np.arange(len(kinds))
    ax.bar(xb, fprs, width=0.62, color=[S.FOURFORM if v > 0.05 else "white" for v in fprs],
           edgecolor=S.FOURFORM, linewidth=0.8)
    ax.axhline(0.05, color=S.INK, ls=(0, (3, 2)), lw=0.7)
    for xi, v in zip(xb, fprs):
        pct = 100 * v
        half = abs(2 * pct - round(2 * pct)) < 1e-9 and abs(pct - round(pct)) > 1e-9
        ax.text(xi, max(v, 0.0) + 0.02 + (0.05 if v <= 0.06 else 0),
                f"{pct:.1f}%" if half else f"{round(pct):.0f}%", ha="center", fontsize=6.5)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("False-positive rate")
    ax.set_xticks(xb, names, fontsize=6.5, rotation=35, ha="right", rotation_mode="anchor")
    S.grid(ax)
    S.panel(ax, letter, "Extrapolation screen, instrument effects")


def coverage_panel(ax, letter: str = "b") -> None:
    """Coverage of the calibrated boundary intervals by sample size."""
    boundary = load("conformal_boundary_intervals.json")
    n_values = [40, 80, 160]
    by = boundary["by_sample_size"]
    cover = np.array([by[str(n)]["coverage"] for n in n_values])
    low = np.array([by[str(n)]["coverage_wilson95"][0] for n in n_values])
    high = np.array([by[str(n)]["coverage_wilson95"][1] for n in n_values])
    ax.errorbar(n_values, cover, yerr=[cover - low, high - cover], color=S.FOURFORM, marker="o", ms=3.5,
                lw=1.2, capsize=2, label="Held-out coverage (Wilson 95% interval)")
    ax.axhline(0.90, color=S.INK, ls=(0, (3, 2)), lw=0.7, label="Nominal 90%")
    ax.set_xscale("log")
    ax.set_xticks(n_values, [str(n) for n in n_values])
    ax.minorticks_off()
    ax.set_xlim(32, 200)
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Points per dataset")
    ax.set_ylabel("Coverage of 90% boundary intervals")
    ov = boundary["overall"]
    ax.text(0.02, 0.97, f"All sizes: {ov['coverage']:.1%} of {boundary['test_datasets']} held-out simulations",
            transform=ax.transAxes, fontsize=6.5, color=S.MUTED, va="top")
    ax.legend(loc="lower right", fontsize=6.5)
    S.grid(ax)
    S.panel(ax, letter, "Boundary intervals calibrated by sample size")


def compact_panel(ax, letter: str = "d") -> None:
    """Compact encoder against random features and the generic-function control encoder."""
    learned = load("pre1900_strict_one_class.json")
    cross_family = load("pre1900_cross_family_conformal.json")
    generic_control = load("general_function_control_evaluation.json")
    syn_t = learned["variants"]["zero_units"]["synthetic"]
    syn_r = learned["variants"]["random_zero_units"]["synthetic"]
    cf = cross_family["synthetic"]
    gc = generic_control["synthetic"]
    groups = [
        ("k-NN distance\n(datasets)", syn_t["knn"]["auroc"], None, syn_r["knn"]["auroc"], None),
        ("Mahalanobis\n(datasets)", syn_t["shrinkage_mahalanobis"]["auroc"], None,
         syn_r["shrinkage_mahalanobis"]["auroc"], None),
        ("Family level", cf["family_auroc"], cf["family_auroc_bootstrap_95_ci"],
         gc["family_knn_auroc"], gc["family_knn_bootstrap_95_ci"]),
    ]
    w = 0.34
    for i, (name, tv, tci, cv, cci) in enumerate(groups):
        for dx, v, ci, face in ((-w / 2 - 0.01, tv, tci, S.LEARNED), (w / 2 + 0.01, cv, cci, "white")):
            ax.bar(i + dx, v, w, color=face, edgecolor=S.LEARNED if face != "white" else S.UNTRAINED,
                   linewidth=0.8)
            if ci:
                ax.errorbar(i + dx, v, yerr=[[v - ci[0]], [ci[1] - v]], color=S.INK, lw=0.7, capsize=1.6)
            top = ci[1] if ci else v
            ax.text(i + dx, top + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.axvline(1.5, ymax=0.8, color=S.LIGHT, lw=0.8)
    ax.axhline(0.5, color=S.INK, ls=(0, (3, 2)), lw=0.7)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=6.5)
    ax.set_ylim(0, 1.38)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("AUROC (12 synthetic departure\nvs 12 classical families)")
    ax.legend(handles=[
        Patch(facecolor=S.LEARNED, edgecolor=S.LEARNED, label="Compact encoder"),
        Patch(facecolor="white", edgecolor=S.UNTRAINED, label="Control"),
    ], loc="upper left", bbox_to_anchor=(0.0, 1.04), ncol=2, fontsize=6.5, handlelength=1.2,
        columnspacing=1.0)
    S.grid(ax)
    S.panel(ax, letter, "Compact encoder and its controls")


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.WIDTH, 64 * S.MM))
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.985, top=0.89, bottom=0.25, wspace=0.34)
    instrument_panel(fig.add_subplot(gs[0, 0]), "a")
    coverage_panel(fig.add_subplot(gs[0, 1]), "b")
    OUT.mkdir(exist_ok=True)
    S.save(fig, OUT / "figure_observation")
    print(OUT / "figure_observation.png")


if __name__ == "__main__":
    main()
