"""Supplementary figure: measured series at the 1899 horizon (sfigure_measured).

Visualization only: scores are read from the evaluation JSON and the point-cloud
archive. Colours follow figure_scripts/epoch_style.py: post-1900 mechanisms
vermillion filled squares, classical or metrological groups blue open circles. Panel e
shows every individual score behind the medians in panel a.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

EVAL = ROOT / "results" / "real_measurement_case_evaluation.json"
CALIB = ROOT / "results" / "instrument_robustness.json"  # extrapolation-score 5% threshold at n = 120 (785 simulations)
ARCHIVE = ROOT / "data" / "real_measurement_case_pointclouds.npz"
OUT = ROOT / "figures" / "sfigure_measured"
BLUE = S.CLASSICAL
ORANGE = S.LATER
GREY = S.MUTED
MM = S.MM

LABEL = {
    "cryogenic_thermal_expansion": "copper thermal expansion",
    "superconducting_flux_creep": "magnetization relaxation",
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
                                xytext=(dx, dy), fontsize=6.5, color=colour,
                                ha=ha, va="bottom", zorder=5)
            extent = label.get_window_extent(renderer=renderer)
            box = (extent.x0 - 1.2, extent.y0 - 1.2, extent.x1 + 1.2, extent.y1 + 1.2)
            if not overlaps(box):
                taken.append(box)
                if (dx, dy) != candidates[0]:  # leader for every displaced label
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
    ("HAHN1", "Copper expansion", "Temperature (K)", "Expansion coefficient", True),
    ("ECKERLE4", "Circular interference", "Wavelength (nm)", "Transmittance", False),
    ("BENNETT6", "Magnetization relaxation", "ln(time / min)", "Magnetization", True),
]


def main() -> None:
    S.apply()
    result = json.loads(EVAL.read_text(encoding="utf-8"))
    archive = np.load(ARCHIVE, allow_pickle=False)
    ids = [str(value) for value in archive["dataset_ids"]]
    groups = result["group_scores"]
    series = result["series_scores"]
    design = result["design"]

    fig = plt.figure(figsize=(S.WIDTH, 180 * MM))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.55, 0.78, 0.5], hspace=0.55,
                            left=0.1, right=0.985, top=0.955, bottom=0.07)
    mid = grid[1].subgridspec(1, 3, wspace=0.62)

    # -- panel a: raw group medians, both screens ------------------------
    ax = fig.add_subplot(grid[0])
    ax.set_xscale("symlog", linthresh=0.05)
    label_items = []
    for name, row in groups.items():
        post = row["horizon_class"] == "post_1900_mechanism"
        x, y = row["statistical_group_score"], row["learned_group_score"]
        ax.scatter(x, y, s=26, marker="s" if post else "o",
                   facecolor=(ORANGE if post else "white"),
                   edgecolor=(ORANGE if post else BLUE), linewidth=0.9, zorder=3)
        label_items.append((x, y, LABEL.get(name, name), ORANGE if post else BLUE))
    ax.set_xlabel("Extrapolation score (symmetric log axis)")
    ax.set_ylabel("Learned score, compact encoder (group median)")
    S.panel(ax, "a", "Sixteen NIST series in 14 mechanism groups")
    ranking = result["ranking"]
    note = ax.text(0.985, 0.78,
                   f"Learned AUROC {ranking['learned_group_auroc']:.2f} [{ranking['learned_group_auroc_95ci'][0]:.2f}, {ranking['learned_group_auroc_95ci'][1]:.2f}]\n"
                   f"Extrapolation AUROC {ranking['statistical_group_auroc']:.2f} [{ranking['statistical_group_auroc_95ci'][0]:.2f}, {ranking['statistical_group_auroc_95ci'][1]:.2f}]",
                   transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=GREY)
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", markerfacecolor=ORANGE, markeredgecolor=ORANGE,
               markersize=5,
               label=f"Post-1900 mechanism ({design['post_1900_mechanism_groups']} groups)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=BLUE,
               markersize=5,
               label=f"Classical or metrological ({design['classical_or_metrology_groups']} groups)"),
    ], loc="upper left", frameon=False, handlelength=1.6)

    ax.set_ylim(-0.07, 0.50)
    ax.set_xlim(-0.004, 90)
    ax.set_xticks([0, 0.1, 1, 10], ["0", "0.1", "1", "10"])
    thr = json.loads(CALIB.read_text(encoding="utf-8"))["thresholds"]["5pct"]  # 3.3913
    ax.plot([thr, thr], [-0.07, 0.268], color=S.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    thr_label = ax.text(thr, 0.272, "Extrapolation-score reference\nthreshold ($n$ = 120)", fontsize=6.5, color=S.MUTED,
                        va="bottom", ha="center", linespacing=1.15)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    line_px = ax.transData.transform([(thr, -0.07), (thr, 0.272)])
    reserved = [ax.get_legend().get_window_extent(renderer).extents,
                note.get_window_extent(renderer).extents,
                thr_label.get_window_extent(renderer).extents,
                (line_px[0, 0] - 1.5, line_px[0, 1], line_px[1, 0] + 1.5, line_px[1, 1])]
    label_items.sort(key=lambda item: -item[1])
    place_labels(ax, fig, label_items, reserved=reserved)

    # -- panels b-d: three measured series ----------------------------------
    for slot, letter, case in zip((mid[0, 0], mid[0, 1], mid[0, 2]), "bcd", PANEL_CASES):
        dataset, title, xlabel, ylabel, post = case
        axis = fig.add_subplot(slot)
        cloud = np.asarray(archive[f"cloud_{ids.index(dataset)}"], float)
        axis.plot(cloud[:, 0], cloud[:, 1], ".", ms=1.6,
                  color=(ORANGE if post else BLUE), alpha=0.85)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        S.panel(axis, letter, title, x=-0.02)
        if dataset == "HAHN1":
            axis.set_xscale("log")
            axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            axis.xaxis.set_minor_formatter(NullFormatter())

    # -- panel e: every individual score -------------------
    axis = fig.add_subplot(grid[2])
    rng = np.random.default_rng(7)
    released = 0
    for row in series:
        post = row["horizon_class"] == "post_1900_mechanism"
        values = [value for value in row["learned_view_scores"] if np.isfinite(value)]
        if not values:
            continue
        released += len(values)
        y = (1 if post else 0) + rng.uniform(-0.17, 0.17, len(values))
        axis.plot(values, y, "s" if post else "o", ms=2.4,
                  markerfacecolor=(ORANGE if post else "none"),
                  markeredgecolor=(ORANGE if post else BLUE),
                  markeredgewidth=0.6, alpha=0.8, ls="")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Classical", "Post-1900"])
    axis.set_ylim(-0.6, 1.6)
    axis.set_xlabel("Learned score, compact encoder")
    S.panel(axis, "e", f"All {released} individual scores", x=-0.02)

    S.save(fig, OUT)
    plt.close(fig)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
