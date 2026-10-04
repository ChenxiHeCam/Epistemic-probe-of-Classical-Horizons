"""Figure 1: testing a knowledge horizon (overview).

a, real dated law families on a time axis: the 51 pre-1900 families of the
   source-verified registry (filled: the 40 that train the compact encoder; open: the 11 held out) and the 55 later families of the benchmark, with the
   1899 horizon. Years are read from data/pre1900_registry.json and
   data/post1900_formula_registry.json.
b, how one relation is tested (schematic): knowledge horizon, admissible pre-horizon model,
   prediction with an observation model, calibrated discrepancy, verdict; the generic screens
   (representation distance, extrapolation) flag relations for testing. The large encoder's data
   design is drawn separately (make_sfig_data_design.py).
c, the three kinds of evidence (schematic; the curves are illustrative): representation screen,
   extrapolation screen, named-prediction test.
d, the verdicts on real data: Solar-System orbits (compatible, a classical
   control), copper heat capacity (localized incompatibility), COBE-FIRAS
   (whole-range incompatibility); abstention is illustrative.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


def plain_log_ticks(*axes):
    """Plain tick labels on log axes (no 70% superscripts)."""
    for a_ in axes:
        a_.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        a_.set_minor_formatter(NullFormatter())

OUT = ROOT / "figures" / "Figure_1_overview"
DARK = "#4A4A4A"


def years(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj if isinstance(obj, list) else next(v for v in obj.values() if isinstance(v, list))
    return np.array([r["first_valid_year"] for r in rows if r.get("first_valid_year") is not None])


def box(ax, x, y, w, h, text, edge=DARK, lw=0.8, bold_first=False, size=6.5, face="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=2, clip_on=False))
    if bold_first and "\n" in text:
        head, body = text.split("\n", 1)
        ax.text(x + w / 2, y + h - 0.03, head, ha="center", va="top", fontsize=size,
                fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 0.03 - 0.07 * (head.count(chr(10)) + 1), body, ha="center", va="top", fontsize=size,
                linespacing=1.2, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                linespacing=1.2, zorder=3)


def arrow(ax, p, q, color=DARK, ls="-", text=None, tx=0.0, ty=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, linewidth=0.7,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=6.5, color=S.MUTED,
                ha="center", va="center")


# ---------------- a: dated law families ----------------
def T(yr):
    yr = np.asarray(yr, float)
    return np.where(yr <= 1900, 0.62 * (yr - 1600) / 300, 0.62 + 0.38 * (yr - 1900) / 100)


def pre1900_years_by_split():
    """First dated year of each pre-1900 family and whether it trains the compact encoder.

    The family split (40 train, 4 validation, 7 test) is the one stored with the compact
    encoder's corpus in data/pre1900_audited_pointclouds.npz.
    """
    laws = json.loads((ROOT / "data" / "pre1900_registry.json").read_text(encoding="utf-8"))["laws"]
    pc = np.load(ROOT / "data" / "pre1900_audited_pointclouds.npz", allow_pickle=True)
    split = {str(f): str(s_) for f, s_ in zip(pc["family_ids"], pc["splits"])}
    yrs = np.array([r["first_valid_year"] for r in laws], float)
    train = np.array([split[r["family_id"]] == "train" for r in laws])
    return yrs, train


def timeline(ax):
    pre, train = pre1900_years_by_split()
    post = years(ROOT / "data" / "post1900_formula_registry.json")
    early = pre < 1600
    ax.set_xlim(-0.04, 1.02)
    ax.set_ylim(-0.22, 1.0)
    ax.axis("off")
    H = float(T(1899.5))
    # pre-1900: filled = training families, open = held out (validation and test); post-1900 squares
    bins = np.floor(pre / 10) * 10
    for b in np.unique(bins[~early]):
        sel = (bins == b) & ~early
        order = np.r_[np.where(sel & train)[0], np.where(sel & ~train)[0]]
        xs = float(T(b + 5))
        for k, idx in enumerate(order):
            ax.scatter([xs], [0.05 + 0.085 * k], s=7, marker="o", linewidths=0.6,
                       facecolor=S.CLASSICAL if train[idx] else "white", edgecolor=S.CLASSICAL, zorder=3)
    pbins = np.floor(post / 5) * 5
    for b in np.unique(pbins):
        n = int((pbins == b).sum())
        ax.scatter([float(T(b + 2.5))] * n, 0.05 + 0.085 * np.arange(n), s=7, marker="s", color=S.LATER,
                   linewidths=0, zorder=3)
    ax.plot([0, 1], [0, 0], color=S.INK, lw=0.6)
    for yr in (1600, 1700, 1800, 1900, 1950, 2000):
        ax.plot([float(T(yr))] * 2, [0, -0.04], color=S.INK, lw=0.5)
        ax.text(float(T(yr)), -0.07, str(yr), ha="center", va="top", fontsize=6.5)
    if early.any():
        ax.annotate("earlier", xy=(0.0, 0.09), xytext=(-0.035, 0.09), fontsize=6.5,
                    color=S.CLASSICAL, ha="right", va="center")
        for k, idx in enumerate(np.where(early)[0]):
            ax.scatter([-0.012], [0.05 + 0.085 * k], s=7, marker="o", linewidths=0.6,
                       facecolor=S.CLASSICAL if train[idx] else "white", edgecolor=S.CLASSICAL)
    ax.plot([H, H], [-0.04, 0.98], color=S.INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(H - 0.006, 0.97, "Knowledge horizon: 1899", fontsize=7, fontweight="bold", ha="right", va="top")
    ax.text(0.0, 0.97, "Pre-1900 law families with dated sources\n(filled: training; open: held out)",
            fontsize=6.5, color=S.CLASSICAL, va="top", ha="left", linespacing=1.2)
    ax.text(1.0, 0.97, "Later law families\n(evaluation only)", fontsize=6.5,
            color=S.LATER, va="top", ha="right", linespacing=1.2)
    S.panel(ax, "a", "Dated law families on either side of the horizon", x=0.0, y=1.0)


# ---------------- b: how one relation is tested (schematic) ----------------
def flow(ax):
    """Knowledge horizon -> admissible pre-horizon model -> prediction -> calibrated discrepancy ->
    verdict, with the generic screens as a side branch that flags relations for testing.

    The vertical axis is in printed points, so text offsets are true sizes."""
    fig = ax.figure
    pos = ax.get_position()
    H = pos.height * fig.get_figheight() * 72
    W = pos.width * fig.get_figwidth() * 72
    ax.set_xlim(0, 1)
    ax.set_ylim(0, H)
    ax.axis("off")

    def rbox(x, y, w, h, edge=DARK, lw=0.8, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={3.0 / W}",
                                    mutation_aspect=W, linewidth=lw, edgecolor=edge,
                                    facecolor="white", linestyle=ls, zorder=2, clip_on=False))

    top, bh = H - 1, 58
    y0 = top - bh
    gap = 0.03
    vw = 0.165
    w = (1 - vw - 4 * gap) / 4
    steps = [
        ("Knowledge\nhorizon", "sources dated\nup to a set year"),
        ("Admissible\npre-horizon model", "a law and its\nparameters, known\nat the horizon"),
        ("Prediction", "with an observation\nmodel: noise and\ninstrument effects"),
        ("Calibrated\ndiscrepancy", "data against a null\nsimulated from\nthe prediction"),
    ]
    xs = [k * (w + gap) for k in range(4)]
    for x, (title, body) in zip(xs, steps):
        rbox(x, y0, w, bh)
        n = 2  # bodies start at the same height in every box
        ax.text(x + w / 2, top - 5, title, ha="center", va="top", fontsize=7, fontweight="bold",
                linespacing=1.1, zorder=3)
        ax.text(x + w / 2, top - 5 - 8.6 * n - 3.5, body, ha="center", va="top", fontsize=6.5,
                linespacing=1.15, zorder=3)
    vx = 1 - vw
    for x in xs[1:] + [vx]:
        arrow(ax, (x - gap, y0 + bh / 2), (x, y0 + bh / 2))
    # verdict box: full panel height, listing the four verdicts
    rbox(vx, 1, vw, top - 1, edge=S.INK, lw=1.0)
    ax.text(vx + vw / 2, top - 5, "Verdict", ha="center", va="top", fontsize=7, fontweight="bold", zorder=3)
    ax.text(vx + vw / 2, top - 20,
            "compatible\n\nlocalized\nincompatibility\n\nwhole-range\nincompatibility\n\nabstention",
            ha="center", va="top", fontsize=6.5, linespacing=1.1, zorder=3)
    # side branch: the generic screens, under the first three steps
    sx = xs[0] + 0.03
    sw = xs[2] + w - 0.03 - sx
    sh = 27
    rbox(sx, 1, sw, sh, edge=S.MUTED, ls=(0, (3, 2)))
    ax.text(sx + sw / 2, 1 + sh - 4.5, "Generic screens", ha="center", va="top", fontsize=7,
            fontweight="bold", zorder=3)
    ax.text(sx + sw * 0.34, 5, "representation distance", ha="center", va="bottom", fontsize=6.5,
            color=S.LEARNED, zorder=3)
    ax.text(sx + sw * 0.76, 5, "extrapolation", ha="center", va="bottom", fontsize=6.5,
            color=S.FOURFORM, zorder=3)
    xa = xs[1] + w / 2
    arrow(ax, (xa, 1 + sh), (xa, y0), ls=(0, (3, 2)))
    ax.text(xa + 0.01, (1 + sh + y0) / 2, "flags relations for testing", fontsize=6.5, color=S.MUTED,
            ha="left", va="center")
    S.panel(ax, "b", "How a relation is tested", x=0.0, y=1.0)


# ---------------- c: three kinds of evidence (schematic) ----------------
def evidence(fig, spec):
    sub = spec.subgridspec(1, 6, width_ratios=[0.62, 1.0, 0.62, 1.0, 0.62, 1.0], wspace=0.12)
    rng = np.random.default_rng(3)
    # representation screen
    ax = fig.add_subplot(sub[0, 0])
    pts = rng.normal(0, 0.55, (60, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=3, color=S.CLASSICAL, alpha=0.8, linewidths=0)
    new = np.array([2.6, 1.4])
    ax.scatter(*new, s=12, marker="s", color=S.LATER, zorder=3)
    near = pts[np.argmin(((pts - new) ** 2).sum(1))]
    ax.annotate("", xy=new, xytext=near, arrowprops=dict(arrowstyle="<->", color=S.LEARNED, lw=0.8))
    ax.set_xlim(-2, 3.4); ax.set_ylim(-2, 2.4)
    frame(ax, S.LEARNED)
    label(fig.add_subplot(sub[0, 1]), "Representation\nscreen",
          "distance from pre-1900\nrelations in a learned\nembedding")
    S.panel(ax, "c", "Three kinds of evidence", x=0.0, y=1.1)
    # extrapolation screen
    ax = fig.add_subplot(sub[0, 2])
    x = np.linspace(0.05, 1, 16)
    xb = 0.58
    y = np.where(x < xb, 0.15 + x, 0.15 + x - 2.4 * np.clip(x - xb, 0, None) ** 1.5)
    ax.axvspan(xb, 1.05, color=S.GRID, lw=0)
    ax.plot([0.03, xb], [0.18, 0.15 + xb], color=S.CLASSICAL, lw=1.0)
    ax.plot([xb, 1.03], [0.15 + xb, 1.18], color=S.CLASSICAL, lw=1.0, ls=(0, (3, 2)))
    ax.scatter(x, y + rng.normal(0, 0.01, x.size), s=3, color=S.DATA, zorder=3)
    ax.axvline(xb, color=S.INK, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlim(0, 1.06); ax.set_ylim(0, 1.25)
    frame(ax, S.FOURFORM)
    label(fig.add_subplot(sub[0, 3]), "Extrapolation\nscreen",
          "fit in the fitted\nregime (left); score\nthe extrapolation\nregion (shaded)")
    # named-prediction test
    ax = fig.add_subplot(sub[0, 4])
    xs = np.linspace(-3.5, 6, 200)
    ax.fill_between(xs, np.exp(-0.5 * xs ** 2), color=S.LIGHT, lw=0)
    ax.plot(xs, np.exp(-0.5 * xs ** 2), color=S.MUTED, lw=0.7)
    ax.axvline(4.6, color=S.INK, lw=1.1)
    ax.set_xlim(-3.6, 6.2); ax.set_ylim(0, 1.2)
    frame(ax, S.INK)
    label(fig.add_subplot(sub[0, 5]), "Named-prediction\ntest",
          "observed statistic\n(line) against a null\nsimulated from the\npre-1900 prediction\n(grey)")


def frame(ax, color):
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(1.0)


def label(ax, name, desc):
    ax.axis("off")
    ax.text(0.0, 1.0, name, transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold",
            linespacing=1.1)
    ax.text(0.0, 0.66, desc, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, linespacing=1.15)


# ---------------- d: verdicts on real data ----------------
def verdicts(fig, spec):
    sub = spec.subgridspec(1, 4, wspace=0.55)
    # compatible: Solar-System orbits
    ax = fig.add_subplot(sub[0, 0])
    a = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07])
    P = np.array([0.241, 0.615, 1.0, 1.881, 11.86, 29.46, 84.01, 164.8])
    sl, ic = np.polyfit(np.log10(a), np.log10(P), 1)
    aa = np.logspace(np.log10(0.3), np.log10(40), 50)
    ax.loglog(aa, 10 ** ic * aa ** sl, color=S.CLASSICAL, lw=1.1, ls=(0, (3, 2)))
    ax.scatter(a, P, s=7, color=S.DATA, zorder=3)
    plain_log_ticks(ax.xaxis, ax.yaxis)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Semi-major axis (AU)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("Period (yr)", fontsize=6.5, labelpad=1)
    verdict(ax, "Compatible", "Solar-System orbits,\nKepler's law")
    S.panel(ax, "d", "Verdicts, shown on real data", x=0.0, y=1.2)
    # localized: copper heat capacity
    ax = fig.add_subplot(sub[0, 1])
    row = json.loads((ROOT / "data" / "nist_cp_coef.json").read_text(encoding="utf-8"))[0]
    coef, (lo, hi) = row[1], row[2]
    Tt = np.logspace(np.log10(max(lo, 4)), np.log10(hi), 160)
    Cp = 10 ** sum(c * np.log10(Tt) ** k for k, c in enumerate(coef))
    r = Cp / np.median(Cp[Tt > 0.7 * hi])
    ax.set_xscale("log")
    _grid = np.logspace(np.log10(lo), np.log10(hi), 120)
    ax.axvspan(np.quantile(_grid, 0.75), hi, color=S.CLASSICAL, alpha=0.08, lw=0)
    ax.plot(Tt, r, color=S.DATA, lw=1.1)
    _lvl = np.exp(json.loads((ROOT / "results" / "theory_clock_experiment.json").read_text(encoding="utf-8"))["cases"][2]["old_horizon"]["parameters"]["log_plateau"]) / np.median(Cp[Tt > 0.7 * hi])
    ax.plot([4, 300], [_lvl, _lvl], color=S.CLASSICAL, lw=1.1, ls=(0, (3, 2)))
    ax.set_ylim(-0.05, 1.25)
    plain_log_ticks(ax.xaxis)
    ax.set_xlabel("Temperature (K)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("$C_p$ / high-$T$ reference", fontsize=6.5, labelpad=1)
    verdict(ax, "Localized incompatibility", "Copper heat capacity,\nDulong–Petit constant")
    # whole-range: FIRAS
    ax = fig.add_subplot(sub[0, 2])
    rows = []
    for line in (ROOT / "data" / "firas_monopole.txt").read_text(encoding="utf-8").splitlines():
        f = line.strip().split()
        if not f or f[0].startswith("#"):
            continue
        try:
            rows.append((float(f[0]), float(f[1])))
        except ValueError:
            pass
    d = np.array(sorted(rows))
    x, y = d[:, 0], d[:, 1]
    low = x < 4
    amp = json.loads((ROOT / "results" / "theory_clock_experiment.json").read_text(encoding="utf-8"))["cases"][0]["old_horizon"]["parameters"]["amplitude"]
    xx = np.linspace(0, x.max(), 100)
    ax.plot(xx, amp * xx ** 2, color=S.CLASSICAL, lw=1.1, ls=(0, (3, 2)))
    ax.scatter(x, y, s=3, color=S.DATA, zorder=3)
    ax.set_ylim(-20, 650)
    ax.set_xlabel("$\\nu/c$ (cm$^{-1}$)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("Intensity (MJy sr$^{-1}$)", fontsize=6.5, labelpad=1)
    verdict(ax, "Whole-range incompatibility", "COBE-FIRAS spectrum,\nRayleigh–Jeans law")
    # abstention: illustrative
    ax = fig.add_subplot(sub[0, 3])
    ax.scatter([0.35, 0.62], [0.42, 0.58], s=9, color=S.DATA)
    ax.text(0.5, 0.82, "?", fontsize=12, ha="center", va="center", color=S.MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    verdict(ax, "Abstention", "too little evidence\n(illustrative)")
    for a_ in fig.axes[-4:]:
        a_.tick_params(labelsize=6.5, length=2, pad=1)


def verdict(ax, name, desc):
    ax.text(0.5, 1.04, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.text(0.5, -0.27, desc, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=S.MUTED,
            linespacing=1.15)


def main():
    S.apply()
    fig = plt.figure(figsize=(S.WIDTH, 172 * S.MM))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.36, 0.68, 0.40, 0.001, 0.72],
                          left=0.075, right=0.985, top=0.968, bottom=0.1, hspace=0.3)
    timeline(fig.add_subplot(gs[0]))
    flow(fig.add_subplot(gs[1]))
    evidence(fig, gs[2])
    verdicts(fig, gs[4])
    S.save(fig, OUT)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
