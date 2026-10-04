"""Save the per-dataset scores behind the synthetic benchmarks of Supplementary Fig. 2.

Two suites, each reproduced by importing the original script unchanged (its module-level code
runs on import, so the suite, seeds and scoring functions are exactly those of the original):

1. engine/ensemble_engine.py, the 36-dataset suite (18 classical relations, 18 departures, seed 3):
   extrapolation-residual ratio, split-conformal nonconformity and their rank average (the
   four-form component). AUROCs with the class-stratified bootstrap of ensemble_engine.ci
   (3000 resamples, seed 0). Also the ROC curve of the rank average, a pointwise bootstrap band
   and the true-positive rate at the threshold giving at most 5% false positives.
   -> results/benchmark_36_suite.json

2. casestudies/baselines_ablation.py, the 16-phenomenon suite (8 classical, 8 later):
   Ramsey RESET, CUSUM, and the extrapolation-residual ratio with the classical law given,
   auto-selected, and with law and interior both auto-selected. The per-phenomenon loop is the
   body of baselines_ablation.score_all with the scores kept; every AUROC is checked against
   score_all itself.
   -> results/baselines_16_suite.json

The script stops (AssertionError) if any value differs from the original printed values.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def quiet_import(name: str, folder: Path):
    sys.path.insert(0, str(folder))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = importlib.import_module(name)
    sys.path.pop(0)
    return mod, buf.getvalue()


def roc_points(y, s):
    """Empirical ROC (score >= threshold flags a departure), thresholds descending."""
    thr = np.unique(s)[::-1]
    fpr = [0.0]
    tpr = [0.0]
    for t in thr:
        tpr.append(float(np.mean(s[y == 1] >= t)))
        fpr.append(float(np.mean(s[y == 0] >= t)))
    return np.array(fpr), np.array(tpr), thr


def tpr_at_fpr(y, s, max_fpr=0.05):
    fpr, tpr, thr = roc_points(y, s)
    ok = fpr[1:] <= max_fpr + 1e-12
    if not ok.any():
        return 0.0, None, 0.0
    i = int(np.argmax(np.where(ok, tpr[1:], -1)))
    return float(tpr[1:][i]), float(thr[i]), float(fpr[1:][i])


def stratified_boot(y, n_boot=3000, seed=0):
    r = np.random.RandomState(seed)
    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]
    for _ in range(n_boot):
        yield np.concatenate([r.choice(neg, len(neg), True), r.choice(pos, len(pos), True)])


# ------------------------------------------------------------------ 36-dataset suite
E, printed36 = quiet_import("ensemble_engine", ROOT / "engine")

# Recompute the unmasked scores on the same suite to recover each dataset's index and check identity.
bd_all = np.array([E.breakdown(x, y) for x, y, _ in E.phen])
cf_all = np.array([E.conformal(x, y) for x, y, _ in E.phen])
lab_all = np.array([l for _, _, l in E.phen])
ok = np.isfinite(bd_all) & np.isfinite(cf_all)
assert np.array_equal(bd_all[ok], E.bd) and np.array_equal(cf_all[ok], E.cf) and np.array_equal(lab_all[ok], E.lab)
idx = np.where(ok)[0]

expected36 = {"extrapolation_residual_ratio": (0.716, 0.534, 0.877),
              "split_conformal": (0.769, 0.599, 0.907),
              "four_form_rank_average": (0.804, 0.639, 0.937)}
scores36 = {"extrapolation_residual_ratio": E.bd, "split_conformal": E.cf, "four_form_rank_average": E.ens}
methods36 = {}
for key, s in scores36.items():
    a, lo, hi = E.ci(s, E.lab)
    got = (round(a, 3), round(lo, 3), round(hi, 3))
    assert got == expected36[key], (key, got, expected36[key])
    methods36[key] = {"auroc": float(a), "ci95": [float(lo), float(hi)]}

y36 = E.lab.astype(int)
s_ens = E.ens.astype(float)
fpr, tpr, thr = roc_points(y36, s_ens)
assert abs(np.trapz(np.r_[tpr, 1.0], np.r_[fpr, 1.0]) - methods36["four_form_rank_average"]["auroc"]) < 1e-9

grid = np.linspace(0, 1, 101)
band = []
tpr5_boot = []
for b in stratified_boot(y36, 3000, 0):
    f_b, t_b, _ = roc_points(y36[b], s_ens[b])
    f_b = np.r_[f_b, 1.0]
    t_b = np.r_[t_b, 1.0]
    # upper envelope of the step curve at each FPR (TPR reached at or below that FPR)
    band.append([t_b[f_b <= g + 1e-12].max() for g in grid])
    tpr5_boot.append(tpr_at_fpr(y36[b], s_ens[b])[0])
band = np.array(band)
tpr5, thr5, fpr5 = tpr_at_fpr(y36, s_ens)

out36 = {
    "source": "engine/ensemble_engine.py (imported unchanged; suite mk(3), 18 classical + 18 departures)",
    "generated_by": "engine/benchmark_suite_scores.py",
    "n_datasets": int(len(y36)),
    "n_classical": int((y36 == 0).sum()),
    "n_departures": int((y36 == 1).sum()),
    "bootstrap": "class-stratified, 3000 resamples, RandomState(0) (ensemble_engine.ci)",
    "printed_by_ensemble_engine": printed36.splitlines()[:4],
    "methods": methods36,
    "datasets": [
        {"index_in_suite": int(i), "label": int(l), "label_name": "departure" if l else "classical",
         "extrapolation_residual_ratio": float(b), "split_conformal": float(c), "four_form_rank_average": float(e)}
        for i, l, b, c, e in zip(idx, y36, E.bd, E.cf, E.ens)
    ],
    "roc_four_form": {
        "fpr": fpr.tolist() + [1.0], "tpr": tpr.tolist() + [1.0],
        "band_fpr_grid": grid.tolist(),
        "band_tpr_lo": np.percentile(band, 2.5, axis=0).tolist(),
        "band_tpr_hi": np.percentile(band, 97.5, axis=0).tolist(),
        "band_method": "pointwise 2.5-97.5 percentiles of the bootstrap ROC curves (same resamples as the AUROC interval)",
        "tpr_at_fpr_le_0.05": tpr5,
        "tpr_at_fpr_le_0.05_ci95": [float(np.percentile(tpr5_boot, 2.5)), float(np.percentile(tpr5_boot, 97.5))],
        "fpr_at_that_threshold": fpr5,
        "threshold_rank_average": thr5,
        "n_departures_flagged": int(round(tpr5 * (y36 == 1).sum())),
    },
}
(RES / "benchmark_36_suite.json").write_text(json.dumps(out36, indent=2))

# ------------------------------------------------------------------ 16-phenomenon suite
B, printed16 = quiet_import("baselines_ablation", ROOT / "casestudies")

conditions = {
    "ratio_law_given": lambda x, y, form, fe: B.breakdown(x, y, form, fe),
    "ratio_law_auto": lambda x, y: B.breakdown_auto(x, y, "low"),
    "ratio_law_and_interior_auto": lambda x, y: B.breakdown_autosplit(x, y),
    "reset": lambda x, y, form, fe: B.reset(x, y, form, fe),
    "cusum": lambda x, y, form, fe: B.cusum(x, y, form, fe),
}
expected16 = {"ratio_law_given": 1.000, "ratio_law_auto": 1.000, "ratio_law_and_interior_auto": 0.969,
              "reset": 0.734, "cusum": 0.672}


def score_all_keep(scorer):
    """Body of baselines_ablation.score_all, unchanged, with names and scores returned."""
    names = []; ys = []; ss = []
    for name, fn, rng, form, lab in B.PH:
        r = np.random.RandomState(3); x = np.sort(r.uniform(*rng, 220)); y = fn(x)
        m = np.isfinite(y) & (np.abs(y) < 1e10); x = x[m]; y = y[m]
        fe = 'high' if name in ('rel energy', 'specific heat') else 'low'
        try: s = scorer(x, y, form, fe) if 'form' in scorer.__code__.co_varnames else scorer(x, y)
        except Exception: s = np.nan
        if np.isfinite(s): names.append(name); ys.append(lab); ss.append(s)
    return names, np.array(ys), np.array(ss, dtype=float)


methods16 = {}
per16 = {}
for key, fn in conditions.items():
    names, ys, ss = score_all_keep(fn)
    a = roc_auc_score(ys, ss)
    assert a == B.score_all(fn), key
    assert round(a, 3) == expected16[key], (key, a, expected16[key])
    boots = [roc_auc_score(ys[b], ss[b]) for b in stratified_boot(ys, 3000, 0)]
    methods16[key] = {"auroc": float(a), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                      "n": int(len(ys))}
    per16[key] = dict(zip(names, ss.tolist()))

labels16 = {name: int(lab) for name, _, _, _, lab in B.PH}
out16 = {
    "source": "casestudies/baselines_ablation.py (imported unchanged; 16 phenomena, RandomState(3), 220 points)",
    "generated_by": "engine/benchmark_suite_scores.py",
    "n_phenomena": len(B.PH),
    "n_classical": int(sum(1 for v in labels16.values() if v == 0)),
    "n_later": int(sum(1 for v in labels16.values() if v == 1)),
    "note": ("The three 'ratio_*' conditions score the extrapolation-residual ratio alone (one of the two "
             "statistics of the four-form component); RESET and CUSUM are computed on the same classical fit."),
    "bootstrap": "class-stratified, 3000 resamples, RandomState(0) (same procedure as ensemble_engine.ci); added here, not printed by the original",
    "printed_by_baselines_ablation": [l for l in printed16.splitlines() if "AUROC" in l or l.strip()[:4] in ("OUR ", "BASE")],
    "methods": methods16,
    "phenomena": [{"name": n, "label": labels16[n], "label_name": "later" if labels16[n] else "classical",
                   **{k: per16[k].get(n) for k in conditions}} for n in labels16],
}
(RES / "baselines_16_suite.json").write_text(json.dumps(out16, indent=2))

print("36-dataset suite (n=%d):" % len(y36))
for k, v in methods36.items():
    print("  %-30s %.3f [%.3f, %.3f]" % (k, v["auroc"], *v["ci95"]))
print("  TPR at FPR<=0.05: %.3f (FPR %.3f, %d of %d departures) bootstrap [%.3f, %.3f]" % (
    tpr5, fpr5, out36["roc_four_form"]["n_departures_flagged"], (y36 == 1).sum(), *out36["roc_four_form"]["tpr_at_fpr_le_0.05_ci95"]))
print("16-phenomenon suite:")
for k, v in methods16.items():
    print("  %-30s %.3f [%.3f, %.3f] n=%d" % (k, v["auroc"], *v["ci95"], v["n"]))
