"""Render the Nature Communications-style EPOCH case-study figure.

The artwork is prepared at 180 mm width with 7 pt Arial lettering, colour-safe
encoding, vector PDF/SVG output and a 600 dpi RGB review PNG.  It combines the
pre-score-frozen learned result, the independent compact/statistical audit and
the separately labelled post-score case-study expansion without changing any
score, threshold or checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
PHASE_A = ROOT / "results" / "phase_a_pre1900_v3_evaluation.json"
RANDOM_A = ROOT / "results" / "random_phase_a_pre1900_v3_evaluation.json"
FROZEN = ROOT / "results" / "frozen_post1900_formula_evaluation.json"
EXPANSION = ROOT / "results" / "case_study_expansion_evaluation.json"
LARGE = ROOT / "results" / "case_study_expansion_v3_evaluation.json"
REGISTRY = ROOT / "data" / "case_study_expansion_registry.json"
ARCHIVE = ROOT / "data" / "case_study_expansion_pointclouds.npz"
OUT_PNG = ROOT / "figures" / "figure_case_study_composite.png"
OUT_PDF = ROOT / "figures" / "figure_case_study_composite.pdf"
OUT_SVG = ROOT / "figures" / "figure_case_study_composite.svg"

# Okabe--Ito colours, with marker shape duplicating the theory-routing code.
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7B61A8"
GREY = "#7A7F83"
LIGHT = "#D9DEE2"
INK = "#1F2326"

THEORY_CRITICAL = {
    "casimir_parallel_plates",
    "parity_beta_asymmetry",
    "pound_rebka_redshift",
    "aspect_polarization_correlation",
    "accelerating_universe_residual",
    "gw150914_leading_chirp",
}

CURVE_CASES = [
    ("d", "cherenkov_threshold", "Cherenkov threshold"),
    ("e", "fano_asymmetric_resonance", "Fano resonance"),
    ("f", "accelerating_universe_residual", "Cosmic acceleration"),
    ("g", "pound_rebka_redshift", "Pound–Rebka"),
]

ABBREVIATIONS = {
    "franck_hertz_excitation": "FH",
    "bec_condensate_fraction": "BEC",
    "cherenkov_threshold": "Ch",
    "casimir_parallel_plates": "Ca",
    "parity_beta_asymmetry": "PV",
    "anderson_localized_envelope": "An",
    "pound_rebka_redshift": "PR",
    "fano_asymmetric_resonance": "Fa",
    "aspect_polarization_correlation": "BA",
    "accelerating_universe_residual": "SN",
    "gw150914_leading_chirp": "GW",
}

CASE_LABELS = {
    "franck_hertz_excitation": "Franck–Hertz",
    "bec_condensate_fraction": "BEC fraction",
    "cherenkov_threshold": "Cherenkov",
    "casimir_parallel_plates": "Casimir",
    "parity_beta_asymmetry": "Parity",
    "anderson_localized_envelope": "Anderson",
    "pound_rebka_redshift": "Pound–Rebka",
    "fano_asymmetric_resonance": "Fano",
    "aspect_polarization_correlation": "Bell–Aspect",
    "accelerating_universe_residual": "Cosmic acceleration",
    "gw150914_leading_chirp": "GW chirp",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax, letter: str) -> None:
    ax.text(-0.15, 1.00, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, fontweight="bold", color=INK, clip_on=False)


def clean_axes(ax, grid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(color="#E5E7E9", linewidth=0.45, zorder=0)
    ax.tick_params(pad=1.5)


def score_values(score_map: dict) -> np.ndarray:
    return np.asarray([row["one_class_ensemble"] for row in score_map.values()], dtype=float)


def learned_roc_panel(ax, trained: dict, random: dict) -> None:
    for payload, color, linestyle, label in (
        (trained, BLUE, "-", "trained  0.925 [0.851–0.982]"),
        (random, GREY, (0, (3, 2)), "unoptimized  0.649"),
    ):
        controls = score_values(payload["sets"]["known_internal"]["scores"])
        later = score_values(payload["sets"]["curated_all_post1900"]["scores"])
        labels = np.r_[np.zeros(controls.size, dtype=int), np.ones(later.size, dtype=int)]
        fpr, tpr, _ = roc_curve(labels, np.r_[controls, later])
        ax.step(fpr, tpr, where="post", color=color, linestyle=linestyle,
                linewidth=1.15, label=label, zorder=2)
    ax.plot([0, 1], [0, 1], color=LIGHT, linestyle=(0, (2, 2)), linewidth=0.65, zorder=1)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           xlabel="False-positive rate", ylabel="True-positive rate")
    ax.text(0.03, 0.96, "Large learned component\n55 later vs 9 pre-1900 groups",
            transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.82})
    ax.legend(frameon=False, loc="lower right", handlelength=2.4, borderpad=0.1)
    clean_axes(ax, grid=True)
    panel_label(ax, "a")


def frozen_roc_arrays(payload: dict, score_key: str) -> tuple[np.ndarray, np.ndarray]:
    controls = np.asarray([row[score_key] for row in payload["heldout_pre1900_controls"]], float)
    later = np.asarray([row[score_key] for row in payload["post1900_families"]], float)
    return controls[np.isfinite(controls)], later[np.isfinite(later)]


def component_roc_panel(ax, payload: dict) -> None:
    primary = payload["primary_family_level"]
    specifications = (
        ("stat_score", "statistical_four_form", ORANGE, "data-anchored  0.888 [0.800–0.959]"),
        ("learned_score", "learned", BLUE, "compact learned  0.669 [0.534–0.798]"),
    )
    for score_key, result_key, color, label in specifications:
        controls, later = frozen_roc_arrays(payload, score_key)
        labels = np.r_[np.zeros(controls.size, dtype=int), np.ones(later.size, dtype=int)]
        fpr, tpr, _ = roc_curve(labels, np.r_[controls, later])
        # Touch the deposited result so a changed JSON schema fails loudly.
        assert np.isfinite(primary[result_key]["auroc_post1900_vs_heldout_pre1900"])
        ax.step(fpr, tpr, where="post", color=color, linewidth=1.15, label=label, zorder=2)
    ax.plot([0, 1], [0, 1], color=LIGHT, linestyle=(0, (2, 2)), linewidth=0.65, zorder=1)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="False-positive rate")
    ax.text(0.03, 0.96, "Independent frozen audit\n55 later vs 11 pre-1900 families",
            transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.82})
    ax.legend(frameon=False, loc="lower right", handlelength=2.4, borderpad=0.1)
    clean_axes(ax, grid=True)
    panel_label(ax, "b")


def score_map_panel(ax, small: dict, large: dict) -> None:
    learned = {row["family_id"]: row["one_class_ensemble"] for row in large["case_rows"]}
    rows = small["case_rows"]
    y_positions = np.arange(len(rows))[::-1]
    for y, row in zip(y_positions, rows):
        family = row["family_id"]
        statistical = row["statistical_percentile_vs_785_size_matched_calibration"]
        learned_value = learned[family]
        ax.plot([learned_value, statistical], [y, y], color=LIGHT, linewidth=0.65, zorder=1)
        ax.scatter(learned_value, y, marker="o", s=13, facecolor=BLUE,
                   edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(statistical, y, marker="s", s=12, facecolor=ORANGE,
                   edgecolor="white", linewidth=0.35, zorder=3)
        if family in THEORY_CRITICAL:
            ax.scatter(1.055, y, marker="^", s=13, facecolor="white",
                       edgecolor=PURPLE, linewidth=0.75, clip_on=False, zorder=3)
    ax.set_yticks(y_positions, [CASE_LABELS[row["family_id"]] for row in rows], fontsize=5.3)
    ax.set(xlim=(-0.02, 1.10), ylim=(-0.7, len(rows) - 0.3),
           xlabel="Continuous component score")
    ax.set_xticks([0, 0.5, 1.0])
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE,
               markeredgecolor="white", color=BLUE, markersize=4, label="L  learned"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor=ORANGE,
               markeredgecolor="white", color=ORANGE, markersize=3.8, label="S  statistical"),
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="white",
               markeredgecolor=PURPLE, color=PURPLE, markersize=4, label="T  required"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=1,
              bbox_to_anchor=(0.01, 0.98), handletextpad=0.25, labelspacing=0.25,
              borderpad=0.0)
    clean_axes(ax, grid=False)
    ax.grid(axis="x", color="#E5E7E9", linewidth=0.45, zorder=0)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "c")


def normalized_curves(observed: np.ndarray, clean: np.ndarray,
                      incumbent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joined = np.r_[observed, clean, incumbent]
    lo, hi = np.quantile(joined, [0.01, 0.99])
    span = max(float(hi - lo), 1e-12)
    return tuple((np.clip(values, lo, hi) - lo) / span
                 for values in (observed, clean, incumbent))


def curve_panel(ax, archive, family: str, letter: str, title: str,
                small_rows: dict, large_rows: dict) -> None:
    family_ids = archive["family_ids"].astype(str)
    index = int(np.flatnonzero(family_ids == family)[0])
    cloud = np.asarray(archive["X"][index], dtype=float)
    x = cloud[:, 0]
    x = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
    observed, successor, incumbent = normalized_curves(
        cloud[:, 1], archive["clean_y"][index], archive["incumbent_y"][index]
    )
    ax.plot(x, incumbent, color=BLUE, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.plot(x, successor, color=ORANGE, linewidth=1.05, zorder=2)
    ax.scatter(x[::5], observed[::5], s=3.5, color=INK, alpha=0.72,
               linewidth=0, zorder=3)
    learned = large_rows[family]["one_class_ensemble"]
    statistical = small_rows[family]["statistical_percentile_vs_785_size_matched_calibration"]
    ax.text(0.03, 0.96, title, transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
    ax.text(0.97, 0.84, f"L {learned:.2f}  S {statistical:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.6, color=GREY,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.82})
    if family in THEORY_CRITICAL:
        ax.text(0.04, 0.08, "T", transform=ax.transAxes, color=PURPLE,
                fontsize=6.0, fontweight="bold")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.06, 1.08))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BFC4C7")
        spine.set_linewidth(0.5)
    panel_label(ax, letter)


def main() -> None:
    style()
    trained = load(PHASE_A)
    random = load(RANDOM_A)
    frozen = load(FROZEN)
    small = load(EXPANSION)
    large = load(LARGE)
    small_rows = {row["family_id"]: row for row in small["case_rows"]}
    large_rows = {row["family_id"]: row for row in large["case_rows"]}
    archive = np.load(ARCHIVE, allow_pickle=False)

    # 180 mm wide: production-scale double-column artwork.
    fig = plt.figure(figsize=(7.087, 4.95), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.18, 0.82], hspace=0.36)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.08], wspace=0.38)
    bottom = outer[1].subgridspec(1, 4, wspace=0.18)

    learned_roc_panel(fig.add_subplot(top[0, 0]), trained, random)
    component_roc_panel(fig.add_subplot(top[0, 1]), frozen)
    score_map_panel(fig.add_subplot(top[0, 2]), small, large)
    for position, (letter, family, title) in enumerate(CURVE_CASES):
        curve_panel(fig.add_subplot(bottom[0, position]), archive, family, letter, title,
                    small_rows, large_rows)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color=INK, markersize=3,
               label="generated points"),
        Line2D([], [], color=ORANGE, linewidth=1.05, label="later relation/simulator"),
        Line2D([], [], color=BLUE, linestyle=(0, (3, 2)), linewidth=0.9,
               label="declared incumbent/comparator"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), handlelength=2.0, columnspacing=1.2)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.095)

    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    # Matplotlib writes RGBA PNGs; Nature Portfolio requests RGB artwork.
    with Image.open(OUT_PNG) as image:
        image.convert("RGB").save(OUT_PNG, dpi=(600, 600))
    fig.savefig(OUT_PDF, facecolor="white")
    fig.savefig(OUT_SVG, facecolor="white")
    print(json.dumps({"png": str(OUT_PNG), "pdf": str(OUT_PDF),
                      "svg": str(OUT_SVG), "width_mm": 180,
                      "png_dpi": 600}, indent=2))


if __name__ == "__main__":
    main()
