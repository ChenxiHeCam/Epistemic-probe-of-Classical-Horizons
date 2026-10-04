"""Figure: historical measurements under the 1899 horizon (manuscript Fig. 2).

Panels a-d are the four measured cases; panel e is Millikan's photoelectric
measurement; panel f compares the goodness-of-fit p-values of the pre-1900 and
successor predictions on the same data. Colours follow figure_scripts/epoch_style.py: measurements black,
pre-1900 (classical) predictions blue and dashed, the successor relation
vermillion. Badges give the extrapolation percentile and the verdict, using the
verdict names of the manuscript.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
import epoch_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

OUT = ROOT / "figures"
BL, RD, GY, K = S.CLASSICAL, S.LATER, S.UNTRAINED, S.DATA
sys.path.insert(0, str(ROOT / "casestudies"))
from theory_clock_experiment import DEBYE_CV  # noqa: E402  (same function the experiment fitted)
from make_component_calibration import pvalue_panel  # noqa: E402

CLOCK = {c["case"]: c for c in json.loads((ROOT / "results" / "theory_clock_experiment.json")
                                          .read_text(encoding="utf-8"))["cases"]}


def pv(case, horizon):
    p_ = CLOCK[case][horizon]["pvalue"]
    return f"p = {p_:.3f}"


def succ(case):
    return CLOCK[case]["advanced_horizon"]["parameters"]
DASH = (0, (4, 2))


def badge(ax, txt, y=0.03):
    ax.text(0.985, y, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, zorder=10,
            color=S.INK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4F4F4", ec="#BDBDBD", lw=0.5))


def leg(ax, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("facecolor", "white")
    kw.setdefault("edgecolor", "none")
    kw.setdefault("framealpha", 0.9)
    kw.setdefault("fontsize", 6.5)
    return ax.legend(**kw)


def load2(fn):
    rows = []
    for line in open(fn):
        s = line.strip()
        if not s or s[0].isalpha() or s.startswith("#"):
            continue
        p = s.replace(",", " ").split()
        try:
            rows.append((float(p[0]), float(p[1])))
        except ValueError:
            pass
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def note(ax, x, y, txt, **kw):
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("color", S.MUTED)
    kw.setdefault("bbox", dict(fc="white", ec="none", pad=0.4))
    kw.setdefault("zorder", 6)
    kw.setdefault("va", "top")
    kw.setdefault("ha", "left")
    kw.setdefault("linespacing", 1.3)
    ax.text(x, y, txt, **kw)


S.apply()
fig = plt.figure(figsize=(S.WIDTH, 146 * S.MM))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.92], left=0.085, right=0.985, top=0.965,
                      bottom=0.13, wspace=0.34, hspace=0.52)
A = fig.add_subplot(gs[0, 0]); B = fig.add_subplot(gs[0, 1])
C = fig.add_subplot(gs[1, 0]); D = fig.add_subplot(gs[1, 1])
E = fig.add_subplot(gs[2, 0]); F = fig.add_subplot(gs[2, 1])
for ax in (A, B, C, D, E):
    S.grid(ax)

# a: FIRAS blackbody
nu, I = load2(ROOT / "data" / "firas_monopole.txt")
o = np.argsort(nu); x = nu[o]; y = I[o]; lo = x < 4
aRJ = CLOCK["FIRAS"]["old_horizon"]["parameters"]["amplitude"]  # the tested Rayleigh-Jeans fit
xx = np.linspace(0, x.max(), 200)
A.plot(xx, aRJ * xx ** 2, color=BL, lw=1.4, ls=DASH,
       label=f"Rayleigh–Jeans prediction ({pv('FIRAS', 'old_horizon')})", zorder=2)
_pf = succ("FIRAS")
_xp = xx[xx > 0]
A.plot(_xp, np.exp(_pf["log_amplitude"]) * _xp ** 3 / np.expm1(_pf["inverse_temperature_scale"] * _xp),
       color=RD, lw=1.2, zorder=4, label=f"Planck prediction ({pv('FIRAS', 'advanced_horizon')})")
A.scatter(x, y, s=7, c=K, zorder=3, label="COBE-FIRAS spectrum")
A.set_ylim(-210, 700); A.set_yticks([0, 200, 400, 600]); A.set_xlim(-0.5, 22)
A.set_xlabel("Frequency $\\nu/c$ (cm$^{-1}$)"); A.set_ylabel("Intensity (MJy sr$^{-1}$)")
leg(A, loc="upper right", bbox_to_anchor=(1.0, 1.03))
badge(A, "Extrapolation percentile 0.95\nWhole-range incompatibility")
ins = A.inset_axes([0.64, 0.44, 0.33, 0.22])
ins.scatter(x, y, s=1.5, c=K, zorder=3); ins.plot(xx[xx > 0], aRJ * xx[xx > 0] ** 2, color=BL, lw=0.8, ls=DASH)
ins.plot(_xp, np.exp(_pf["log_amplitude"]) * _xp ** 3 / np.expm1(_pf["inverse_temperature_scale"] * _xp), color=RD, lw=0.8)
ins.set_yscale("log"); ins.set_ylim(1, 1e5); ins.set_xlim(0, 22)
ins.set_yticks([1e0, 1e4]); ins.set_xticks([10, 20])
ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
ins.minorticks_off()
ins.tick_params(labelsize=6.5, length=1.8, width=0.5, pad=1.2)
for s_ in ins.spines.values():
    s_.set_linewidth(0.5)
S.panel(A, "a", "Cosmic microwave background spectrum")

# b: Bertozzi
KE = np.array([.5, 1, 1.5, 4.5, 15]); b2 = np.array([.752, .828, .922, .974, 1.0])
kk = np.linspace(.25, 15, 80)
B.plot(kk, 2 * kk / 0.511, color=BL, lw=1.4, ls=DASH,
       label=f"Newtonian prediction ({pv('Bertozzi', 'old_horizon')})", zorder=2)
B.plot(kk, 1 - 1 / (1 + kk / 0.511) ** 2, color=RD, lw=1.2, zorder=2,
       label=f"Relativistic prediction ({pv('Bertozzi', 'advanced_horizon')})")
B.scatter(KE, b2, s=14, c=K, zorder=3, label="Bertozzi 1964")
B.axhline(1, color=GY, ls=(0, (1, 2)), lw=0.8, zorder=1)
note(B, 6.2, 1.1, "$\\beta^{2}=1$ (speed of light)", va="bottom")
B.set_ylim(0, 3.45); B.set_xlim(-0.4, 15.6)
note(B, 1.9, 2.4, "Newtonian $\\beta^{2}$ reaches 58.7\n($v = 7.7c$) at 15 MeV")
B.set_xlabel("Kinetic energy (MeV)"); B.set_ylabel("$\\beta^{2}=v^{2}/c^{2}$")
leg(B, loc="upper right", bbox_to_anchor=(1.0, 1.03))
badge(B, "Extrapolation percentile 0.66\nWhole-range incompatibility")
S.panel(B, "b", "Electron speed and kinetic energy")

# c: specific heat
C.set_xscale("log")
mats = json.load(open(ROOT / "data" / "nist_cp_coef.json"))


def cp(coef, T):
    L = np.log10(T)
    return 10 ** np.clip(sum(cc * L ** k for k, cc in enumerate(coef)), -30, 30)


grey_done = False
for row in mats:
    name, coef, (lo_, hi_) = row[0], row[1], row[2]
    T = np.logspace(np.log10(max(lo_, 4)), np.log10(hi_), 160)
    Cp = cp(coef, T)
    r = Cp / np.median(Cp[T > 0.7 * hi_])
    if name.startswith("Copper"):
        cu_norm = np.median(Cp[T > 0.7 * hi_])
        # interior used by the tested constant fit: upper quartile of the experiment's grid
        _grid = np.logspace(np.log10(lo_), np.log10(hi_), 120)
        C.axvspan(np.quantile(_grid, 0.75), hi_, color=S.CLASSICAL, alpha=0.08, lw=0)
        C.plot(T, r, color=K, lw=1.4, zorder=4, label="Copper (NIST reference curve)")
        _pc = succ("specific_heat_Cu")
        _deb = _pc["plateau"] * DEBYE_CV(_pc["theta_K"] / T) + _pc["gamma"] * T
        C.plot(T, _deb / np.median(Cp[T > 0.7 * hi_]), color=RD, lw=1.2, zorder=5,
               label=f"Debye plus electronic ({pv('specific_heat_Cu', 'advanced_horizon')})")
    else:
        C.plot(T, r, color=GY, lw=0.7, alpha=0.7, zorder=2,
               label=None if grey_done else "Five other NIST reference curves")
        grey_done = True
_lvl = np.exp(CLOCK["specific_heat_Cu"]["old_horizon"]["parameters"]["log_plateau"]) / cu_norm
C.plot([4, 300], [_lvl, _lvl], color=BL, lw=1.4, ls=DASH, zorder=3,
       label=f"Dulong–Petit constant ({pv('specific_heat_Cu', 'old_horizon')})")
C.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v)); C.xaxis.set_minor_formatter(NullFormatter())
C.set_ylim(-0.66, 2.3); C.set_yticks([0, 0.5, 1.0]); C.set_xlim(3.6, 340)
C.set_xlabel("Temperature $T$ (K)"); C.set_ylabel("$C_p$ / high-$T$ reference")
leg(C, loc="upper left", bbox_to_anchor=(0.0, 1.03), frameon=False)
badge(C, "Copper: extrapolation percentile 0.95\nLocalized incompatibility")
S.panel(C, "c", "Heat capacity of solids")

# d: Onnes 1911
T = np.array([4.00, 4.10, 4.15, 4.19, 4.21, 4.25, 4.30, 4.35, 4.40])
R = np.array([np.nan, np.nan, np.nan, np.nan, 0.110, 0.118, 0.126, 0.134, 0.142])
norm = T > 4.20; Rtc = 0.11
D.axvspan(4.0, 4.20, color=GY, alpha=0.10, lw=0)
D.axvline(4.20, ymax=0.74, color=S.INK, ls=(0, (1, 1.5)), lw=0.8, zorder=1)
_old = CLOCK["Onnes"]["old_horizon"]["parameters"]
_ta = np.linspace(4.0, 4.42, 20)
D.plot(_ta, _old["slope"] * _ta + _old["intercept"], color=BL, lw=1.4, ls=DASH, zorder=2,
       label=f"Smooth normal-state trend ({pv('Onnes', 'old_horizon')})")
_po = succ("Onnes")
_tn = np.linspace(4.20, 4.42, 20)
D.plot([4.0, 4.20, 4.20], [0.0, 0.0, _po["normal_slope"] * 4.20 + _po["normal_intercept"]], color=RD, lw=1.2,
       zorder=1, label=f"Superconducting transition ({pv('Onnes', 'advanced_horizon')})")
D.plot(_tn, _po["normal_slope"] * _tn + _po["normal_intercept"], color=RD, lw=1.2, zorder=1)
D.errorbar(T[norm], R[norm], yerr=0.006, fmt="o", ms=3, c=K, elinewidth=0.7, capsize=1.6, zorder=3,
           label="Onnes 1911, mercury")
D.scatter(T[~norm], np.full((~norm).sum(), 0.003), marker="v", s=14, c=K, zorder=3)
note(D, 4.012, 0.009, "$R<10^{-5}\\,\\Omega$", va="bottom", color=S.INK, bbox=dict(fc="#F3F3F3", ec="none", pad=0.4))
note(D, 4.012, 0.072, "Transition at 4.20 K;\n$R$ falls by $>10^{4}$", bbox=dict(fc="#F3F3F3", ec="none", pad=0.4))
D.set_ylim(-0.010, 0.215); D.set_yticks([0, 0.04, 0.08, 0.12, 0.16]); D.set_xlim(3.985, 4.435)
D.set_xlabel("Temperature $T$ (K)"); D.set_ylabel("Resistance $R$ ($\\Omega$)")
leg(D, loc="upper left", bbox_to_anchor=(-0.005, 1.03))
badge(D, "Extrapolation\npercentile 0.66\nLocalized incompatibility", y=0.2)
S.panel(D, "d", "Mercury resistance near 4.2 K")

# e: Millikan photoelectric
# the digitized observations and the fits stored by the advancing-horizon experiment
nus = np.array([5.49, 7.41, 8.21, 9.59]) * 1e14
V = np.array([0.45, 1.25, 1.58, 2.14])
_pm = succ("Millikan")
slope, icpt = _pm["slope_V_per_1e14Hz"], _pm["intercept_V"]
nu0 = -icpt / slope * 1e14
level = CLOCK["Millikan"]["old_horizon"]["parameters"]["constant_V"]
xx = np.linspace(nu0, 10e14, 50)
E.axhline(0, color=GY, ls=(0, (1, 2)), lw=0.8, zorder=1)
E.plot(xx / 1e14, slope * xx / 1e14 + icpt, color=RD, lw=1.4, zorder=2,
       label=f"Photoelectric relation ({pv('Millikan', 'advanced_horizon')})")
E.hlines(level, 4.0, 10.0, color=BL, lw=1.4, ls=DASH, zorder=2,
         label=f"Frequency-independent energy ({pv('Millikan', 'old_horizon')})")
E.scatter(nus / 1e14, V, s=14, c=K, zorder=4, label="Millikan 1916, sodium")
E.plot([nu0 / 1e14], [0], "D", mfc="white", mec=S.INK, ms=4, zorder=5)
E.annotate("$\\nu_0$", xy=(nu0 / 1e14, 0.02), xytext=(4.12, 0.6), color=S.MUTED,
           fontsize=6.5, ha="left", va="bottom",
           arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=0.7, shrinkA=2, shrinkB=3))
E.set_xlim(4.0, 10.05); E.set_ylim(-1.3, 4.4)
E.set_xticks([4, 6, 8, 10]); E.set_yticks([-1, 0, 1, 2, 3])
E.set_xlabel("Frequency $\\nu$ ($10^{14}$ Hz)"); E.set_ylabel("Stopping potential $V_{\\rm stop}$ (V)")
leg(E, loc="upper left", bbox_to_anchor=(0.0, 1.03))
badge(E, "Slope recovers $h/e$ to 0.3%\nNamed-prediction test\nWhole-range incompatibility")
S.panel(E, "e", "Photoelectric stopping potential", x=-0.02)

# f: the same measurements under the pre-1900 and successor predictions
pvalue_panel(F, "f", legend="below")

OUT.mkdir(exist_ok=True)
S.save(fig, OUT / "figure_cases")
print(OUT / "figure_cases.png")
