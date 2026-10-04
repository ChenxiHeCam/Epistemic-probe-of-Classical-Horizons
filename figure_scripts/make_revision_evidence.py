"""Create the evidence-summary figure (figures/figure_revision_evidence.png and .pdf)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#7A7F87"
INK = "#1A1A1A"
BG = "#FBFBF9"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                     "axes.labelsize": 9, "axes.linewidth": 0.8})

clock = json.loads((RESULTS / "theory_clock_experiment.json").read_text(encoding="utf-8"))
physical_controls = json.loads(
    (RESULTS / "physical_irrelevant_horizon_controls.json").read_text(encoding="utf-8")
)
robust = json.loads((RESULTS / "instrument_robustness.json").read_text(encoding="utf-8"))
boundary = json.loads((RESULTS / "conformal_boundary_intervals.json").read_text(encoding="utf-8"))
learned = json.loads((RESULTS / "pre1900_strict_one_class.json").read_text(encoding="utf-8"))
cross_family = json.loads((RESULTS / "pre1900_cross_family_conformal.json").read_text(encoding="utf-8"))
generic_control = json.loads((RESULTS / "general_function_control_evaluation.json").read_text(encoding="utf-8"))
strict_registry = json.loads((RESULTS / "pre1900_registry_population_report.json").read_text(encoding="utf-8"))
corpus = json.loads((RESULTS / "classical_corpus_provenance_audit.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.7), facecolor=BG)
for ax in axes.flat:
    ax.set_facecolor(BG)

# a: clock advancement.
ax = axes[0, 0]
names = [row["case"].replace("specific_heat_Cu", "Cu heat capacity") for row in clock["cases"]]
old_p = np.array([row["old_horizon"]["pvalue"] for row in clock["cases"]])
new_p = np.array([row["advanced_horizon"]["pvalue"] for row in clock["cases"]])
physical_p = np.array([row["pvalue"] for row in physical_controls["cases"]])
y = np.arange(len(names))
for yi, left, right in zip(y, old_p, new_p):
    ax.plot([left, right], [yi, yi], color="#C9CDD2", lw=2, zorder=1)
ax.scatter(old_p, y - 0.12, color=ORANGE, s=44, label="old horizon", zorder=3)
ax.scatter(new_p, y, facecolors="white", edgecolors=BLUE, linewidths=1.8, s=52, label="advanced horizon", zorder=3)
ax.scatter(physical_p, y + 0.12, color=GREY, marker="x", linewidths=1.5, s=42,
           label="wrong-mechanism physical form", zorder=3)
ax.axvline(0.05, color=INK, ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlim(5e-4, 1.3)
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlabel("counterfactual-null p-value (log scale)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.set_title("a   Same data, advanced knowledge horizon", loc="left", weight="bold")
ax.text(0.05, -0.23, "Wrong-mechanism forms are retrospective and all remain rejected; Cu's simple Debye model also fails.",
        transform=ax.transAxes, fontsize=7.8, color=GREY)
ax.spines[["top", "right"]].set_visible(False)

# b: instrument hard negatives at maximum severity.
ax = axes[0, 1]
preferred = ["calibration_drift", "sensor_saturation", "censoring", "x_axis_error", "outliers",
             "correlated_noise", "heteroscedasticity"]
labels = ["drift", "saturation", "censoring", "x error", "outliers", "correlated", "heteroscedastic"]
fprs = []
for kind in preferred:
    candidates = [row for row in robust["summary"] if row["corruption"] == kind]
    row = max(candidates, key=lambda item: item["severity"])
    fprs.append(row["fpr_at_nominal_5pct"])
bars = ax.bar(np.arange(len(labels)), fprs, color=[ORANGE if value > 0.05 else BLUE for value in fprs], width=0.68)
ax.axhline(0.05, color=INK, ls="--", lw=1, label="nominal 5%")
ax.set_ylim(0, 0.72)
ax.set_ylabel("false-positive rate")
ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
for rect, value in zip(bars, fprs):
    ax.text(rect.get_x() + rect.get_width() / 2, value + 0.018, f"{value:.0%}", ha="center", fontsize=8)
ax.set_title("b   Instrument-induced hard negatives", loc="left", weight="bold")
ax.spines[["top", "right"]].set_visible(False)

# c: boundary coverage.
ax = axes[1, 0]
n_values = [40, 80, 160]
cover = [boundary["by_sample_size"][str(n)]["coverage"] for n in n_values]
low = [boundary["by_sample_size"][str(n)]["coverage_wilson95"][0] for n in n_values]
high = [boundary["by_sample_size"][str(n)]["coverage_wilson95"][1] for n in n_values]
ax.errorbar(n_values, cover, yerr=[np.array(cover) - low, np.array(high) - cover], color=BLUE,
            marker="o", lw=2, capsize=4, label="held-out coverage")
ax.axhline(0.90, color=INK, ls="--", lw=1, label="nominal 90%")
ax.set_ylim(0.75, 1.0)
ax.set_xticks(n_values)
ax.set_xlabel("points per dataset")
ax.set_ylabel("boundary-interval coverage")
ax.set_title("c   Size-conditional conformal boundary intervals", loc="left", weight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.text(0.02, 0.04, f"overall: {boundary['overall']['coverage']:.1%} on {boundary['test_datasets']} held-out simulations",
        transform=ax.transAxes, color=GREY, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# d: source-audited learned evidence.
ax = axes[1, 1]
labels = ["strict\nkNN", "strict\nMahalanobis", "strict\nfamily", "generic\nfamily", "random\nkNN"]
values = [learned["variants"]["zero_units"]["synthetic"]["knn"]["auroc"],
          learned["variants"]["zero_units"]["synthetic"]["shrinkage_mahalanobis"]["auroc"],
          cross_family["synthetic"]["family_auroc"],
          generic_control["synthetic"]["family_knn_auroc"],
          learned["variants"]["random_zero_units"]["synthetic"]["knn"]["auroc"]]
colors = [BLUE, BLUE, ORANGE, GREY, "#BFC3C7"]
bars = ax.bar(np.arange(5), values, color=colors, width=0.62)
ax.axhline(0.5, color=INK, ls="--", lw=1)
ax.set_ylim(0.15, 1.15)
ax.set_ylabel("AUROC")
ax.set_xticks(np.arange(5), labels, fontsize=8)
for rect, value in zip(bars, values):
    ax.text(rect.get_x() + rect.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", fontsize=8)
ax.set_title("d   Learned-component audit", loc="left", weight="bold")
ax.text(0.02, 0.965,
        f"Strict rebuild: {strict_registry['admitted_records']} point clouds, "
        f"{strict_registry['admitted_families']} cited families; no breakdowns in training.\n"
        "Frozen family threshold: 2/12 breakdown; 0/12 synthetic and 0/7 cited classical flagged.\n"
        "Matched generic family AUROC 0.38; strict-control difference 0.53 [0.26, 0.76].",
        transform=ax.transAxes, color=BLUE, fontsize=7.7, va="top",
        bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.92, "pad": 2.5})
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("EPOCH evidence audit: what survives strict testing", x=0.06, y=0.985,
             ha="left", fontsize=15, weight="bold", color=INK)
fig.text(0.06, 0.95,
         "Theory-conditioned horizon updates and strict classical-only transfer are promising; real-data error control remains limiting.",
         fontsize=9.5, color=GREY)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.89, hspace=0.42, wspace=0.35)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "figure_revision_evidence.png", dpi=240, facecolor=BG, bbox_inches="tight")
fig.savefig(OUT / "figure_revision_evidence.pdf", facecolor=BG, bbox_inches="tight")
manuscript_out = ROOT / "manuscript" / "figures"
manuscript_out.mkdir(exist_ok=True)
fig.savefig(manuscript_out / "figure_revision_evidence.png", dpi=240, facecolor=BG, bbox_inches="tight")
print(OUT / "figure_revision_evidence.png")
