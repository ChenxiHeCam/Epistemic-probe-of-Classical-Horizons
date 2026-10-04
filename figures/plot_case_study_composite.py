"""Figure: temporal transfer and component scores across later physics (manuscript Fig. 2).

Visualization only: every score is read from the evaluation JSON files and the
point-cloud archive. Colours and terms follow figure_scripts/epoch_style.py and
the manuscript: representation screen (purple), extrapolation screen (green), later
relation (vermillion), pre-1900 prediction (blue, dashed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PHASE_A = ROOT / "results" / "phase_a_pre1900_v3_evaluation.json"
RANDOM_A = ROOT / "results" / "random_phase_a_pre1900_v3_evaluation.json"
FROZEN = ROOT / "results" / "frozen_post1900_formula_evaluation.json"
EXPANSION = ROOT / "results" / "case_study_expansion_evaluation.json"
LARGE = ROOT / "results" / "case_study_expansion_v3_evaluation.json"
ARCHIVE = ROOT / "data" / "case_study_expansion_pointclouds.npz"
OUT = ROOT / "figures" / "figure_case_study_composite"

THEORY_CONDITIONED = {
    "casimir_parallel_plates",
    "parity_beta_asymmetry",
    "pound_rebka_redshift",
    "aspect_polarization_correlation",
    "accelerating_universe_residual",
    "gw150914_leading_chirp",
}

CURVE_CASES = [
    ("d", "cherenkov_threshold", "Cherenkov emission"),
    ("e", "fano_asymmetric_resonance", "Fano resonance"),
    ("f", "accelerating_universe_residual", "Cosmic acceleration"),
    ("g", "pound_rebka_redshift", "Pound–Rebka redshift"),
]

CASE_LABELS = {
    "franck_hertz_excitation": "Franck–Hertz",
    "bec_condensate_fraction": "Condensate fraction",
    "cherenkov_threshold": "Cherenkov threshold",
    "casimir_parallel_plates": "Casimir plates",
    "parity_beta_asymmetry": "Parity asymmetry",
    "anderson_localized_envelope": "Anderson envelope",
    "pound_rebka_redshift": "Pound–Rebka",
    "fano_asymmetric_resonance": "Fano resonance",
    "aspect_polarization_correlation": "Bell–Aspect",
    "accelerating_universe_residual": "Cosmic acceleration",
    "gw150914_leading_chirp": "Gravitational-wave chirp",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_values(score_map: dict) -> np.ndarray:
    return np.asarray([row["one_class_ensemble"] for row in score_map.values()], dtype=float)


def roc(ax, controls: np.ndarray, later: np.ndarray, **kw) -> None:
    labels = np.r_[np.zeros(controls.size, dtype=int), np.ones(later.size, dtype=int)]
    fpr, tpr, _ = roc_curve(labels, np.r_[controls, later])
    ax.step(fpr, tpr, where="post", linewidth=1.2, zorder=2, **kw)


def roc_axes(ax, xlabel: str) -> None:
    ax.plot([0, 1], [0, 1], color=S.LIGHT, linewidth=0.7, zorder=1)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel=xlabel,
           ylabel="True-positive rate (55 later families)")
    ax.legend(loc="lower right", handlelength=2.2, borderaxespad=0.2, fontsize=6.5, frameon=True,
              facecolor="white", edgecolor="none", framealpha=1.0)
    S.grid(ax, "both")


def learned_roc_panel(ax, trained: dict, random: dict) -> None:
    res = trained["comparisons"]["known_internal_vs_curated_all_post1900"]
    lo, hi = res["ci95"]
    rnd = random["comparisons"]["known_internal_vs_curated_all_post1900"]["auroc"]
    for payload, color, ls, label in (
        (trained, S.LEARNED, "-", f"Trained encoder\nAUROC {res['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"),
        (random, S.UNTRAINED, (0, (3, 2)), f"Untrained network\nAUROC {rnd:.3f}"),
    ):
        roc(ax, score_values(payload["sets"]["known_internal"]["scores"]),
            score_values(payload["sets"]["curated_all_post1900"]["scores"]),
            color=color, linestyle=ls, label=label)
    roc_axes(ax, "False-positive rate (nine pre-1900 groups)")
    S.panel(ax, "a", "Large encoder")


def frozen_roc_arrays(payload: dict, score_key: str) -> tuple[np.ndarray, np.ndarray]:
    controls = np.asarray([row[score_key] for row in payload["heldout_pre1900_controls"]], float)
    later = np.asarray([row[score_key] for row in payload["post1900_families"]], float)
    return controls[np.isfinite(controls)], later[np.isfinite(later)]


def component_roc_panel(ax, payload: dict) -> None:
    primary = payload["primary_family_level"]
    for score_key, result_key, color, ls, name in (
        ("stat_score", "statistical_four_form", S.FOURFORM, "-", "Extrapolation screen"),
        ("learned_score", "learned", S.LEARNED, (0, (4, 1.5)), "Compact encoder"),
    ):
        auc = primary[result_key]["auroc_post1900_vs_heldout_pre1900"]
        lo, hi = primary[result_key]["family_bootstrap_95_ci"]
        controls, later = frozen_roc_arrays(payload, score_key)
        roc(ax, controls, later, color=color, linestyle=ls, label=f"{name}\nAUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
    roc_axes(ax, "False-positive rate (11 pre-1900 families)")
    S.panel(ax, "b", "Extrapolation screen and compact encoder")


def score_map_panel(ax, small: dict, large: dict) -> None:
    learned = {row["family_id"]: row["one_class_ensemble"] for row in large["case_rows"]}
    rows = small["case_rows"]
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        fam = row["family_id"]
        four = row["statistical_percentile_vs_785_size_matched_calibration"]
        lv = learned[fam]
        ax.plot([lv, four], [y, y], color=S.LIGHT, linewidth=0.7, zorder=1)
        ax.scatter(lv, y + 0.12, marker="o", s=14, facecolor=S.LEARNED, edgecolor="white", linewidth=0.3, zorder=3)
        ax.scatter(four, y - 0.12, marker="s", s=13, facecolor=S.FOURFORM, edgecolor="white", linewidth=0.3, zorder=3)
        if fam in THEORY_CONDITIONED:
            ax.scatter(1.07, y, marker="^", s=14, facecolor="white", edgecolor=S.INK,
                       linewidth=0.7, clip_on=False, zorder=3)
    ax.set_yticks(ys, [CASE_LABELS[r["family_id"]] for r in rows], fontsize=6.5)
    ax.set(xlim=(-0.02, 1.12), ylim=(-0.7, len(rows) - 0.3), xlabel="Score")
    ax.set_xticks([0, 0.5, 1.0])
    handles = [
        Line2D([], [], marker="o", ls="none", mfc=S.LEARNED, mec="white", ms=4, label="Learned score (large encoder)"),
        Line2D([], [], marker="s", ls="none", mfc=S.FOURFORM, mec="white", ms=3.8, label="Extrapolation score (percentile)"),
        Line2D([], [], marker="^", ls="none", mfc="white", mec=S.INK, ms=4, label="Assigned to the\nnamed-prediction test"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, columnspacing=0.8,
              handletextpad=0.2, labelspacing=0.25, fontsize=6.5)
    ax.grid(axis="x", color=S.GRID, linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    S.panel(ax, "c", "Eleven further cases", x=-0.02)


def normalized_curves(observed, clean, incumbent):
    joined = np.r_[observed, clean, incumbent]
    lo, hi = np.quantile(joined, [0.01, 0.99])
    span = max(float(hi - lo), 1e-12)
    return tuple((np.clip(v, lo, hi) - lo) / span for v in (observed, clean, incumbent))


def curve_panel(ax, archive, family: str, letter: str, title: str, small_rows: dict,
                large_rows: dict, first: bool) -> None:
    ids = archive["family_ids"].astype(str)
    i = int(np.flatnonzero(ids == family)[0])
    cloud = np.asarray(archive["X"][i], dtype=float)
    x = cloud[:, 0]
    x = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
    observed, successor, incumbent = normalized_curves(cloud[:, 1], archive["clean_y"][i],
                                                       archive["incumbent_y"][i])
    ax.plot(x, incumbent, color=S.CLASSICAL, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.plot(x, successor, color=S.LATER, linewidth=1.1, zorder=2)
    ax.scatter(x[::5], observed[::5], s=3.5, color=S.DATA, alpha=0.75, linewidth=0, zorder=3)
    lv = large_rows[family]["one_class_ensemble"]
    fv = small_rows[family]["statistical_percentile_vs_785_size_matched_calibration"]
    note = f"Learned {lv:.3f}\nExtrapolation {fv:.3f}"
    if family in THEORY_CONDITIONED:
        note += "\nNamed-prediction test"
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.06, 1.08))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(note, fontsize=6.5, labelpad=3, color=S.MUTED, linespacing=1.15, va="top")
    if first:
        ax.set_ylabel("y (scaled)", fontsize=6.5, labelpad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color("#B5B5B5")
        spine.set_linewidth(0.5)
    S.panel(ax, letter, title, x=0.0)


def forest_panel(ax, trained: dict, random: dict, frozen: dict) -> None:
    """Every principal AUROC with its 95% interval, read from the result files."""
    res = ROOT / "results"
    matched = load(res / "matched_pre1900_domain_control_trained_v3.json")
    matched_r = load(res / "matched_pre1900_domain_control_random_v3.json")
    cross = load(res / "pre1900_cross_family_conformal.json")["synthetic"]
    generic = load(res / "general_function_control_evaluation.json")["synthetic"]
    measured = load(res / "real_measurement_case_evaluation.json")["ranking"]
    fam = frozen["primary_family_level"]

    def c(src, key):
        r = src["comparisons"][key]
        return r["auroc"], r["ci95"]

    groups = [
        ("Large encoder vs nine pre-1900 groups", [
            ("38 families 1901–1950, original controls",
             (S.LEARNED, *c(trained, "known_internal_vs_curated_1901_1950")),
             (S.UNTRAINED, *c(random, "known_internal_vs_curated_1901_1950"))),
            ("55 later families, original controls",
             (S.LEARNED, *c(trained, "known_internal_vs_curated_all_post1900")),
             (S.UNTRAINED, *c(random, "known_internal_vs_curated_all_post1900"))),
            ("55 later families, regenerated controls",
             (S.LEARNED, *c(matched, "matched_pre1900_vs_curated_all_post1900")),
             (S.UNTRAINED, *c(matched_r, "matched_pre1900_vs_curated_all_post1900"))),
            ("Same-generator set, 1901–1950",
             (S.LEARNED, *c(trained, "pre1900_internal_vs_generated_1901_1950")),
             (S.UNTRAINED, *c(random, "pre1900_internal_vs_generated_1901_1950"))),
        ]),
        ("55 later vs 11 held-out pre-1900 families", [
            ("Extrapolation screen",
             (S.FOURFORM, fam["statistical_four_form"]["auroc_post1900_vs_heldout_pre1900"],
              fam["statistical_four_form"]["family_bootstrap_95_ci"]), None),
            ("Compact encoder",
             (S.LEARNED, fam["learned"]["auroc_post1900_vs_heldout_pre1900"],
              fam["learned"]["family_bootstrap_95_ci"]), None),
        ]),
        ("12 synthetic departure vs 12 classical families", [
            ("Compact vs generic-function encoder",
             (S.LEARNED, cross["family_auroc"], cross["family_auroc_bootstrap_95_ci"]),
             (S.UNTRAINED, generic["family_knn_auroc"], generic["family_knn_bootstrap_95_ci"])),
        ]),
        ("Measured NIST series, 5 vs 9 groups", [
            ("Learned score (compact encoder)",
             (S.LEARNED, measured["learned_group_auroc"], measured["learned_group_auroc_95ci"]), None),
            ("Extrapolation score",
             (S.FOURFORM, measured["statistical_group_auroc"], measured["statistical_group_auroc_95ci"]), None),
        ]),
    ]
    y = 0.0
    ticks, labels = [], []
    for head, rows in groups:
        ax.text(-0.62, y, head, fontsize=6.5, fontweight="bold", va="center", ha="left",
                transform=ax.get_yaxis_transform())
        y -= 1.0
        for label, main, ctrl in rows:
            col, est, (lo, hi) = main
            ax.plot([lo, hi], [y + 0.12, y + 0.12], color=col, lw=1.1, solid_capstyle="butt")
            ax.plot(est, y + 0.12, "o", ms=3.6, color=col)
            text = f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
            if ctrl:
                ccol, cest, (clo, chi) = ctrl
                ax.plot([clo, chi], [y - 0.22, y - 0.22], color=ccol, lw=0.9, solid_capstyle="butt")
                ax.plot(cest, y - 0.22, "o", ms=3.4, mfc="white", mec=ccol, mew=0.8)
                text += f"   control {cest:.3f}"
            ax.text(1.02, y, text, fontsize=6.5, va="center", ha="left", transform=ax.get_yaxis_transform())
            ticks.append(y)
            labels.append(label)
            y -= 1.0
        y -= 0.25
    ax.axvline(0.5, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ticks, labels, fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(0.1, 1.0)
    ax.set_xticks([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.5", "0.6", "0.8", "1.0"])
    ax.set_xlabel("AUROC with 95% interval (chance 0.5)")
    ax.spines["left"].set_visible(False)
    S.grid(ax, "x")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="-", color=S.LEARNED, ms=3.4, lw=1.0, label="Trained encoder"),
        Line2D([], [], marker="o", ls="-", color=S.FOURFORM, ms=3.4, lw=1.0, label="Extrapolation screen"),
        Line2D([], [], marker="o", ls="-", color=S.UNTRAINED, mfc="white", ms=3.4, lw=0.9,
               label="Untrained or control"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.5, borderaxespad=0.2)
    S.panel(ax, "a", "Every principal AUROC that has a bootstrap interval", x=-0.62, y=1.13)


def main() -> None:
    S.apply()
    trained, random, frozen = load(PHASE_A), load(RANDOM_A), load(FROZEN)
    small, large = load(EXPANSION), load(LARGE)
    small_rows = {row["family_id"]: row for row in small["case_rows"]}
    large_rows = {row["family_id"]: row for row in large["case_rows"]}
    archive = np.load(ARCHIVE, allow_pickle=False)

    fig = plt.figure(figsize=(S.WIDTH, 172 * S.MM))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 0.72], hspace=0.62,
                             left=0.07, right=0.985, top=0.95, bottom=0.115)
    forest = outer[0].subgridspec(1, 3, width_ratios=[0.6, 1.0, 0.68], wspace=0.0)
    forest_panel(fig.add_subplot(forest[0, 1]), trained, random, frozen)
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55)
    component_roc_panel(fig.add_subplot(mid[0, 0]), frozen)
    score_map_panel(fig.add_subplot(mid[0, 1]), small, large)
    bottom = outer[2].subgridspec(1, 4, wspace=0.16)
    for k, (letter, family, title) in enumerate(CURVE_CASES):
        curve_panel(fig.add_subplot(bottom[0, k]), archive, family, letter, title,
                    small_rows, large_rows, first=(k == 0))
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=S.DATA, ms=3, label="Generated points"),
        Line2D([], [], color=S.LATER, linewidth=1.1, label="Later relation"),
        Line2D([], [], color=S.CLASSICAL, linestyle=(0, (3, 2)), linewidth=0.9, label="Pre-1900 prediction"),
    ], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.6)
    S.save(fig, OUT)
    png = OUT.with_suffix(".png")
    with Image.open(png) as im:
        im.convert("RGB").save(png, dpi=(600, 600))
    print(png)


if __name__ == "__main__":
    main()
