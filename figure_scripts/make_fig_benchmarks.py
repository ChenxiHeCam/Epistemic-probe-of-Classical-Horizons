"""Supplementary Fig. 2: measured classical controls and the synthetic benchmarks.

a-c  Independently measured classical relations (Solar-System planets, Galilean moons, Boyle 1662),
     each with the power law fitted by least squares in log-log space (the exponent is computed here
     from the data). No screening score is drawn: the 0.02 / 0.00 / 0.38 values have no result file and
     engine/combine_final_v3.py, rerun, does not reproduce them.
d    AUROC with class-stratified bootstrap 95% intervals on the 36-dataset suite of
     engine/ensemble_engine.py, read from results/benchmark_36_suite.json.
e    The 16-phenomenon suite of casestudies/baselines_ablation.py: RESET and CUSUM against the
     extrapolation-residual ratio under three information conditions, read from
     results/baselines_16_suite.json.
f    ROC curve of the extrapolation screen (four-form rank average) on the 36-dataset suite, its pointwise bootstrap band,
     and the true-positive rate at the threshold giving at most 5% false positives, all read from
     results/benchmark_36_suite.json.

No plotted number is typed into this script; both JSON files are written by
engine/benchmark_suite_scores.py, which reproduces the original scripts' printed values.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

OUT = ROOT / "figures"
RES = ROOT / "results"
BL, GR, GY, K = S.CLASSICAL, S.FOURFORM, S.UNTRAINED, S.DATA
B36 = json.loads((RES / "benchmark_36_suite.json").read_text())
B16 = json.loads((RES / "baselines_16_suite.json").read_text())
FIT = dict(color=BL, lw=1.4, zorder=2)
PTS = dict(s=14, c=K, zorder=3, marker="o", linewidths=0)
LIGHT_GREEN = "#CCEBDF"


def leg(ax, **kw):
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("handlelength", 1.4)
    kw.setdefault("handletextpad", 0.5)
    kw.setdefault("labelspacing", 0.3)
    return ax.legend(**kw)



def ptlabel(ax, names, xs, ys, dx=4, dy=-7):
    for n, x0, y0 in zip(names, xs, ys):
        ax.annotate(n, (x0, y0), textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color=S.MUTED)


S.apply()
fig = plt.figure(figsize=(S.WIDTH, 118 * S.MM))
gs = fig.add_gridspec(2, 1, left=0.085, right=0.985, top=0.945, bottom=0.105, hspace=0.55)
g1 = gs[0].subgridspec(1, 3, wspace=0.42)
g2 = gs[1].subgridspec(1, 3, wspace=0.36, width_ratios=[1.2, 1.5, 0.85])
axA, axB, axC = (fig.add_subplot(g1[0, i]) for i in range(3))
axD, axE, axF = (fig.add_subplot(g2[0, i]) for i in range(3))

# ---- a: Solar-System planets ----------------------------------------------------------------
nm = ["Me", "V", "E", "Ma", "J", "S", "U", "N"]
a = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07])
P = np.array([0.241, 0.615, 1.0, 1.881, 11.86, 29.46, 84.01, 164.8])
sl, ic = np.polyfit(np.log10(a), np.log10(P), 1)
A = axA
A.set_xscale("log"); A.set_yscale("log")
aa = np.logspace(np.log10(0.3), np.log10(40), 50)
A.plot(aa, 10 ** ic * aa ** sl, **FIT, label="Fit, $P\\propto a^{%.3f}$" % sl)
A.scatter(a, P, **PTS, label="Planets (measured)")
ptlabel(A, nm, a, P)
A.set_xlim(0.28, 45); A.set_ylim(0.13, 520)
for _axis in (A.xaxis, A.yaxis):  # plain tick labels, no 70% superscripts
    _axis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    _axis.set_minor_formatter(NullFormatter())
A.set_xlabel("Semi-major axis $a$ (AU)"); A.set_ylabel("Period $P$ (yr)")
leg(A, loc="upper left", borderaxespad=0.1)
S.panel(A, "a", "Solar-System orbits")

# ---- b: Galilean moons ----------------------------------------------------------------------
mn = ["Io", "Eu", "Ga", "Ca"]
ga = np.array([421.8, 671.1, 1070.4, 1882.7]); gP = np.array([1.769, 3.551, 7.155, 16.689])
s2, i2 = np.polyfit(np.log10(ga), np.log10(gP), 1)
B = axB
B.set_xscale("log"); B.set_yscale("log")
gg = np.logspace(np.log10(380), np.log10(2100), 50)
B.plot(gg, 10 ** i2 * gg ** s2, **FIT, label="Fit, $P\\propto a^{%.3f}$" % s2)
B.scatter(ga, gP, **PTS, label="Moons (measured)")
ptlabel(B, mn, ga, gP)
B.set_xlim(370, 2500); B.set_ylim(1.1, 34)
B.xaxis.set_major_locator(FixedLocator([500, 1000, 2000]))
B.xaxis.set_minor_locator(NullLocator())
B.set_xticklabels(["0.5", "1", "2"])
B.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
B.yaxis.set_minor_formatter(NullFormatter())
B.set_yticklabels(["2", "5", "10", "20"])
B.set_xlabel("Semi-major axis $a$ ($10^{6}$ km)"); B.set_ylabel("Period $P$ (days)")
leg(B, loc="upper left", borderaxespad=0.1)
S.panel(B, "b", "Galilean moons of Jupiter")

# ---- c: Boyle 1662 --------------------------------------------------------------------------
bo = np.array(json.load(open(ROOT / "data" / "boyle_1662.json"))); bv = bo[:, 0]; bp = bo[:, 1]
s3, i3 = np.polyfit(np.log10(bv), np.log10(bp), 1)
C = axC
vv = np.linspace(11, 50, 80)
C.plot(vv, 10 ** i3 * vv ** s3, **FIT, label="Fit, $P\\propto V^{%.3f}$" % s3)
C.scatter(bv, bp, **dict(PTS, s=10), label="Boyle 1662 (25 points)")
C.set_xlim(9, 51); C.set_ylim(20, 152); C.set_yticks([20, 60, 100, 140])
C.set_xlabel("Volume (arbitrary units)"); C.set_ylabel("Pressure (inches Hg)")
leg(C, loc="upper right", borderaxespad=0.1)
S.panel(C, "c", "Boyle's 1662 table")

for ax in (axA, axB, axC):
    S.grid(ax, "both")


def bars(ax, labels, entries, colors, hatches, ylim, ylab):
    xp = np.arange(len(entries))
    vals = np.array([e["auroc"] for e in entries])
    lo = np.array([e["ci95"][0] for e in entries]); hi = np.array([e["ci95"][1] for e in entries])
    for i in xp:
        ax.bar(i, vals[i], width=0.62, color=colors[i], edgecolor=GR if hatches[i] else "none",
               hatch=hatches[i], lw=0.6, zorder=2)
    ax.errorbar(xp, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=S.INK, elinewidth=0.7,
                capsize=1.8, zorder=3)
    for i in xp:
        ax.text(i, hi[i] + 0.012, "%.2f" % vals[i], ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0.5, color=S.MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.set_ylim(*ylim); ax.set_xlim(-0.6, len(entries) - 0.4)
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylab)
    S.grid(ax)


plt.rcParams["hatch.linewidth"] = 0.6

# ---- d: 36-dataset suite --------------------------------------------------------------------
m36 = B36["methods"]
D = axD
bars(D, ["Residual\nratio", "Split\nconformal", "Extrapolation\nscreen"],
     [m36["extrapolation_residual_ratio"], m36["split_conformal"], m36["four_form_rank_average"]],
     [LIGHT_GREEN, LIGHT_GREEN, GR], ["////", "////", None], (0.4, 1.07),
     "AUROC (%d datasets)" % B36["n_datasets"])
D.set_yticks([0.4, 0.6, 0.8, 1.0])
S.panel(D, "d", "Synthetic suite")

# ---- e: 16-phenomenon suite -----------------------------------------------------------------
m16 = B16["methods"]
E = axE
bars(E, ["RESET", "CUSUM", "Law\ngiven", "Law\nselected", "Law and\nfitted\nregime\nselected"],
     [m16["reset"], m16["cusum"], m16["ratio_law_given"], m16["ratio_law_auto"],
      m16["ratio_law_and_interior_auto"]],
     [GY, GY, LIGHT_GREEN, LIGHT_GREEN, LIGHT_GREEN], [None, None, "////", "////", "////"], (0.3, 1.2),
     "AUROC (%d phenomena)" % B16["n_phenomena"])
E.set_yticks([0.3, 0.5, 0.7, 0.9])
E.axvline(1.5, color=S.LIGHT, lw=0.6, zorder=1)
E.plot([1.7, 4.3], [1.105, 1.105], color=S.MUTED, lw=0.6, zorder=2)
E.text(3.0, 1.115, "Extrapolation-residual ratio", ha="center", va="bottom", fontsize=6.5, color=S.MUTED)
S.panel(E, "e", "Misspecification tests")

# ---- f: ROC of the extrapolation screen (four-form rank average) ---------------------------------------------------
roc = B36["roc_four_form"]
F = axF
F.fill_between(roc["band_fpr_grid"], roc["band_tpr_lo"], roc["band_tpr_hi"], step="post", color=GR,
               alpha=0.16, lw=0, label="Bootstrap 95% band")
F.plot([0, 1], [0, 1], color=S.MUTED, ls=(0, (1, 2)), lw=0.7, zorder=1)
F.plot(roc["fpr"], roc["tpr"], color=GR, lw=1.3, zorder=3,
       label="Extrapolation screen")
t5 = roc["tpr_at_fpr_le_0.05"]; f5 = roc["fpr_at_that_threshold"]
F.plot([f5], [t5], "o", mfc="white", mec=S.INK, mew=0.8, ms=4, zorder=5)
n_dep = B36["n_departures"]
_m = m36["four_form_rank_average"]
F.text(0.98, 0.20, "AUROC %.2f\n[%.2f, %.2f]" % (_m["auroc"], *_m["ci95"]), ha="right", va="bottom", fontsize=6.5,
       color=S.INK, linespacing=1.25)
F.annotate("FPR \u2264 5%%: TPR %.0f%%\n(%d of %d departures)" % (100 * t5, roc["n_departures_flagged"], n_dep),
           xy=(f5 + 0.012, t5 + 0.004), xytext=(0.98, 0.015), fontsize=6.5, color=S.INK, ha="right", va="bottom",
           linespacing=1.25, arrowprops=dict(arrowstyle="->", color=S.INK, lw=0.6, shrinkA=1, shrinkB=2))
F.set_xlim(-0.01, 1.0); F.set_ylim(-0.01, 1.02)
F.set_xticks([0, 0.5, 1.0]); F.set_yticks([0, 0.5, 1.0])
F.set_xlabel("False-positive rate"); F.set_ylabel("True-positive rate")
S.panel(F, "f", "Extrapolation screen")

OUT.mkdir(exist_ok=True)
S.save(fig, OUT / "figure_benchmarks")
print(OUT / "figure_benchmarks.png")
