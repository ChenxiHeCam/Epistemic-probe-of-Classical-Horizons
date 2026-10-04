"""Formula screening figure (manuscript Fig. 3, two panels) and the two Supplementary figures that
carry the remaining encoder panels (sfigure_encoder_details) and the generated cases.

The panels are drawn by the functions in figures/plot_learned_pre1900_v3.py,
figures/plot_case_study_composite.py and figure_scripts/make_component_calibration.py; this script only
arranges them and assigns panel letters. Every number is read from the results JSON files.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from make_component_calibration import compact_panel  # noqa: E402

OUT = ROOT / "figures"


def module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = module(ROOT / "figures" / "plot_learned_pre1900_v3.py")
C = module(ROOT / "figures" / "plot_case_study_composite.py")

# the panel functions fix their own letters; reassign them by panel title for each figure
_panel = S.panel
LETTERS: dict[str, str] = {}


def _relettered(ax, s, title=None, **kw):
    _panel(ax, LETTERS.get(title, s), title, **kw)


S.panel = _relettered


def screening_figure() -> None:
    """Fig. 3: trained against untrained network by control set, and the extrapolation screen's ROC."""
    LETTERS.clear()
    LETTERS.update({"Trained and untrained network by control set": "a",
                    "Extrapolation screen and compact encoder": "b"})
    phase_a, random_a = L.load(L.PHASE_A), L.load(L.RANDOM_PHASE_A)
    matched, matched_random = L.load(L.MATCHED_PHASE_A), L.load(L.MATCHED_RANDOM_PHASE_A)
    fig = plt.figure(figsize=(S.WIDTH, 84 * S.MM))
    gs = fig.add_gridspec(1, 2, left=0.15, right=0.98, top=0.91, bottom=0.305, wspace=0.34)
    L.comparison_panel(fig.add_subplot(gs[0, 0]), phase_a, random_a, matched, matched_random)
    C.component_roc_panel(fig.add_subplot(gs[0, 1]), C.load(C.FROZEN))
    S.save(fig, OUT / "figure_screening")
    print(OUT / "figure_screening.png")


def encoder_details_figure() -> None:
    """Supplementary figure: large-encoder ROC curves and score distributions, the compact encoder
    against its controls, and the four distance components."""
    LETTERS.clear()
    LETTERS.update({"Ranking of later law families": "a", "Learned scores by group": "b",
                    "Compact encoder and its controls": "c", "The four distance components": "d"})
    phase_a, random_a = L.load(L.PHASE_A), L.load(L.RANDOM_PHASE_A)
    matched = L.load(L.MATCHED_PHASE_A)
    fig = plt.figure(figsize=(S.WIDTH, 160 * S.MM))
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.985, top=0.96, bottom=0.165, hspace=0.48, wspace=0.36)
    L.roc_panel(fig.add_subplot(gs[0, 0]), phase_a, matched)
    L.distribution_panel(fig.add_subplot(gs[0, 1]), phase_a, matched)
    compact_panel(fig.add_subplot(gs[1, 0]), "c")
    L.component_panel(fig.add_subplot(gs[1, 1]), phase_a, random_a)
    S.save(fig, OUT / "sfigure_encoder_details")
    print(OUT / "sfigure_encoder_details.png")


def generated_cases_figure() -> None:
    """Supplementary figure: every principal AUROC with an interval and the eleven further cases."""
    LETTERS.clear()
    LETTERS.update({"Every principal AUROC that has a bootstrap interval": "a", "Eleven further cases": "b"})
    trained, random, frozen = C.load(C.PHASE_A), C.load(C.RANDOM_A), C.load(C.FROZEN)
    small, large = C.load(C.EXPANSION), C.load(C.LARGE)
    small_rows = {row["family_id"]: row for row in small["case_rows"]}
    large_rows = {row["family_id"]: row for row in large["case_rows"]}
    archive = np.load(C.ARCHIVE, allow_pickle=False)
    fig = plt.figure(figsize=(S.WIDTH, 172 * S.MM))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 0.72], hspace=0.62,
                             left=0.07, right=0.985, top=0.95, bottom=0.115)
    forest = outer[0].subgridspec(1, 3, width_ratios=[0.6, 1.0, 0.68], wspace=0.0)
    C.forest_panel(fig.add_subplot(forest[0, 1]), trained, random, frozen)
    mid = outer[1].subgridspec(1, 3, width_ratios=[0.42, 1.0, 0.18], wspace=0.0)
    C.score_map_panel(fig.add_subplot(mid[0, 1]), small, large)
    bottom = outer[2].subgridspec(1, 4, wspace=0.16)
    for k, ((_, family, title), letter) in enumerate(zip(C.CURVE_CASES, "cdef")):
        C.curve_panel(fig.add_subplot(bottom[0, k]), archive, family, letter, title,
                      small_rows, large_rows, first=(k == 0))
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=S.DATA, ms=3, label="Generated points"),
        Line2D([], [], color=S.LATER, linewidth=1.1, label="Later relation"),
        Line2D([], [], color=S.CLASSICAL, linestyle=(0, (3, 2)), linewidth=0.9, label="Pre-1900 prediction"),
    ], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.6)
    S.save(fig, OUT / "sfigure_generated_cases")
    print(OUT / "sfigure_generated_cases.png")


if __name__ == "__main__":
    S.apply()
    screening_figure()
    encoder_details_figure()
    generated_cases_figure()
