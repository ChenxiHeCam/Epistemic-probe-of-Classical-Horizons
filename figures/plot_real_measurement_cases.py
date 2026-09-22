"""Render the real-measurement case-study figure.

Artwork is prepared at 180 mm width with 7 pt Arial lettering, Okabe-Ito
colour-safe encoding duplicated by marker shape, vector PDF/SVG output and a
600 dpi RGB review PNG.  It reads the frozen forward-inference result and
changes no score or checkpoint.

The figure shows raw component scores.  No alert threshold is drawn and no
detection count is plotted; panel e releases every individual measurement-view
score behind the medians in panel a.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "real_measurement_case_evaluation.json"
ARCHIVE = ROOT / "data" / "real_measurement_case_pointclouds.npz"
OUT_PNG = ROOT / "figures" / "figure_real_measurement_cases.png"
OUT_PDF = ROOT / "figures" / "figure_real_measurement_cases.pdf"
OUT_SVG = ROOT / "figures" / "figure_real_measurement_cases.svg"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#666666"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

LABEL = {
    "cryogenic_thermal_expansion": "Cu thermal expansion",
    "superconducting_flux_creep": "flux creep",
    "bose_gas_density": "Bose gas density",
    "rydberg_series": "Rydberg series",
    "semiconductor_magnetotransport": "magnetotransport",
    "semiconductor_mobility": "carrier mobility",
    "ultrasonic_attenuation": "ultrasonic attenuation",
    "circular_interference": "circular interference",
    "load_cell_elasticity": "load-cell elasticity",
    "linewidth_standard": "line-width standard",
    "radiographic_defect_calibration": "radiographic calibration",
    "optical_linewidth_calibration": "optical calibration",
    "polymer_impedance": "polymer impedance",
    "standard_resistor_drift": "resistor drift",
}

def place_labels(ax, fig, items, reserved=()):
    """Greedy non-overlapping label placement with leader lines.

    items: (x, y, text, colour) in data coordinates, placed in the given order.
    Candidate offsets are tried nearest first; the first that clears every
    marker, every already-placed label and every reserved box is kept.  A leader
    line is drawn whenever a label had to move away from its own point.
    """
    candidates = [(5, -2), (5, -7), (-5, -2), (-5, -7), (0, 5), (0, -11),
                  (11, 4), (11, -12), (-11, 4), (-11, -12),
                  (20, 0), (-20, 0), (0, 12), (0, -18),
                  (26, 9), (-26, 9), (26, -15), (-26, -15),
                  (0, 19), (0, -25), (34, -2), (-34, -2),
                  (40, 14), (-40, 14), (40, -20), (-40, -20)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = [tuple(box) for box in reserved]
    for x, y, _, _ in items:
        px, py = ax.transData.transform((x, y))
        taken.append((px - 4.0, py - 4.0, px + 4.0, py + 4.0))

    def overlaps(box):
        x0, y0, x1, y1 = box
        return any(not (x1 < bx0 or bx1 < x0 or y1 < by0 or by1 < y0)
                   for bx0, by0, bx1, by1 in taken)

    for x, y, text, colour in items:
        for dx, dy in candidates:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            label = ax.annotate(text, (x, y), textcoords="offset points",
                                xytext=(dx, dy), fontsize=5, color=colour,
                                ha=ha, va="bottom", zorder=5)
            extent = label.get_window_extent(renderer=renderer)
            box = (extent.x0 - 1.2, extent.y0 - 1.2, extent.x1 + 1.2, extent.y1 + 1.2)
            if not overlaps(box):
                taken.append(box)
                if abs(dx) > 12 or abs(dy) > 12:
                    if dx > 0:
                        anchor = (extent.x0, (extent.y0 + extent.y1) / 2)
                    elif dx < 0:
                        anchor = (extent.x1, (extent.y0 + extent.y1) / 2)
                    else:
                        anchor = ((extent.x0 + extent.x1) / 2,
                                  extent.y0 if dy > 0 else extent.y1)
                    ax_x, ax_y = ax.transData.inverted().transform(anchor)
                    ax.plot([x, ax_x], [y, ax_y], color=colour, lw=0.35,
                            alpha=0.55, zorder=2, solid_capstyle="butt")
                break
            label.remove()


PANEL_CASES = [
    ("HAHN1", "Cu thermal expansion", "temperature (K)", "expansion coefficient", True),
    ("ECKERLE4", "circular interference", "wavelength (nm)", "transmittance", False),
    ("BENNETT6", "superconducting flux creep", "log time (ln min)", "magnetization", True),
]


def main() -> None:
    result = json.loads(EVAL.read_text(encoding="utf-8"))
    archive = np.load(ARCHIVE, allow_pickle=False)
    ids = [str(value) for value in archive["dataset_ids"]]
    groups = result["group_scores"]
    series = result["series_scores"]
    design = result["design"]

    fig = plt.figure(figsize=(180 * MM, 78 * MM))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 1.0],
                            height_ratios=[1.0, 1.0], wspace=0.46, hspace=0.66,
                            left=0.065, right=0.985, top=0.90, bottom=0.115)

    # -- panel a: raw group medians, both components ------------------------
    ax = fig.add_subplot(grid[:, 0])
    ax.set_xscale("symlog", linthresh=0.05)
    label_items = []
    for name, row in groups.items():
        post = row["horizon_class"] == "post_1900_mechanism"
        x, y = row["statistical_group_score"], row["learned_group_score"]
        ax.scatter(x, y, s=26, marker="o" if post else "s",
                   facecolor=(ORANGE if post else "white"),
                   edgecolor=(ORANGE if post else BLUE), linewidth=0.9, zorder=3)
        label_items.append((x, y, LABEL.get(name, name), ORANGE if post else BLUE))
    ax.set_xlabel("statistical score (size-matched)")
    ax.set_ylabel("learned one-class score")
    ax.set_title("a  Real NIST measurements, frozen components",
                 loc="left", fontweight="bold")
    ranking = result["ranking"]
    note = ax.text(0.985, 0.78,
                   f"learned AUROC {ranking['learned_group_auroc']:.2f}\n"
                   f"statistical AUROC {ranking['statistical_group_auroc']:.2f}",
                   transform=ax.transAxes, ha="right", va="top", fontsize=6, color=GREY)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=ORANGE, markeredgecolor=ORANGE,
               markersize=5,
               label=f"post-1900 mechanism ({design['post_1900_mechanism_groups']} groups)"),
        Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=BLUE,
               markersize=5,
               label=f"classical or metrology ({design['classical_or_metrology_groups']} groups)"),
    ], loc="upper left", frameon=False, handlelength=1.6)

    ax.set_ylim(-0.03, 0.50)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reserved = [ax.get_legend().get_window_extent(renderer).extents,
                note.get_window_extent(renderer).extents]
    label_items.sort(key=lambda item: -item[1])
    place_labels(ax, fig, label_items, reserved=reserved)

    # -- panels b-d: three measured series ----------------------------------
    for slot, letter, case in zip((grid[0, 1], grid[0, 2], grid[1, 1]), "bcd", PANEL_CASES):
        dataset, title, xlabel, ylabel, post = case
        axis = fig.add_subplot(slot)
        cloud = np.asarray(archive[f"cloud_{ids.index(dataset)}"], float)
        axis.plot(cloud[:, 0], cloud[:, 1], ".", ms=1.6,
                  color=(ORANGE if post else BLUE), alpha=0.85)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
        if dataset == "HAHN1":
            axis.set_xscale("log")

    # -- panel e: every individual measurement-view score -------------------
    axis = fig.add_subplot(grid[1, 2])
    rng = np.random.default_rng(7)
    released = 0
    for row in series:
        post = row["horizon_class"] == "post_1900_mechanism"
        values = [value for value in row["learned_view_scores"] if np.isfinite(value)]
        if not values:
            continue
        released += len(values)
        y = (1 if post else 0) + rng.uniform(-0.17, 0.17, len(values))
        axis.plot(values, y, "o" if post else "s", ms=2.4,
                  markerfacecolor=(ORANGE if post else "none"),
                  markeredgecolor=(ORANGE if post else BLUE),
                  markeredgewidth=0.6, alpha=0.8, ls="")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["classical", "post-1900"])
    axis.set_ylim(-0.55, 1.55)
    axis.set_xlabel("learned one-class score")
    axis.set_title(f"e  All {released} view scores", loc="left", fontweight="bold")

    for path, kwargs in ((OUT_PDF, {}), (OUT_SVG, {}), (OUT_PNG, {"dpi": 600})):
        fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_PNG.name}, {OUT_PDF.name}, {OUT_SVG.name}")


if __name__ == "__main__":
    main()
