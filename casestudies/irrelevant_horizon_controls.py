"""Wrong-shape controls for the EPOCH clock-advance experiment.

These are not claims about historically proposed theories.  They ask a narrower
diagnostic question: does merely adding a low-dimensional distractor family make
the old-horizon lack of fit disappear?  Each distractor is declared before its
bootstrap null is evaluated and uses the same grid/error model as the paired
clock experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from theory_clock_experiment import (
    ROOT, HorizonFit, bootstrap_gof, cases, bh_qvalues,
)


def weighted_linear(x, y, sigma):
    design = np.column_stack([np.ones_like(x), x])
    weighted = design / sigma[:, None]
    params, *_ = np.linalg.lstsq(weighted, y / sigma, rcond=None)
    return design @ params, params


def firas_distractor(x, y, sigma):
    pred, params = weighted_linear(x, y, sigma)
    return HorizonFit("affine distractor", pred, lambda z: firas_distractor(x, z, sigma),
                      {"intercept": float(params[0]), "slope": float(params[1])},
                      "Two-parameter monotone/affine wrong-shape control.")


def bertozzi_distractor(x, y, sigma):
    weight = 1 / sigma**2
    basis = x**2
    amplitude = np.sum(weight * basis * y) / np.sum(weight * basis**2)
    pred = amplitude * basis
    return HorizonFit("quadratic-through-origin distractor", pred,
                      lambda z: bertozzi_distractor(x, z, sigma),
                      {"amplitude": float(amplitude)}, "Non-saturating wrong-shape control.")


def cp_distractor(x, y, sigma):
    # Power law: parsimonious and flexible on log-log axes, but not a Debye successor.
    design = np.column_stack([np.ones_like(x), np.log(x)])
    params, *_ = np.linalg.lstsq(design / sigma[:, None], np.log(y) / sigma, rcond=None)
    pred = np.exp(design @ params)
    return HorizonFit("power-law distractor", pred, lambda z: cp_distractor(x, z, sigma),
                      {"log_amplitude": float(params[0]), "exponent": float(params[1])},
                      "Two-parameter global power-law wrong-shape control.")


def onnes_distractor(x, y, sigma):
    params = np.polyfit(x, y, 2, w=1 / sigma)
    pred = np.clip(np.polyval(params, x), 1e-8, None)
    return HorizonFit("smooth-quadratic distractor", pred, lambda z: onnes_distractor(x, z, sigma),
                      {"quadratic": [float(v) for v in params]},
                      "Three-parameter smooth control cannot encode a discontinuity.")


def millikan_distractor(x, y, sigma):
    def model(xv, log_a, log_b, c):
        return np.exp(log_a) / (xv + np.exp(log_b)) + c
    params, _ = curve_fit(model, x, y, p0=[0.0, 0.0, np.mean(y)], sigma=sigma,
                          absolute_sigma=True, maxfev=20000)
    pred = model(x, *params)
    return HorizonFit("inverse-frequency distractor", pred, lambda z: millikan_distractor(x, z, sigma),
                      {"log_amplitude": float(params[0]), "log_offset": float(params[1]),
                       "constant": float(params[2])},
                      "Constrained decreasing inverse-frequency wrong-shape control.")


DISTRACTORS = {
    "FIRAS": firas_distractor,
    "Bertozzi": bertozzi_distractor,
    "specific_heat_Cu": cp_distractor,
    "Onnes": onnes_distractor,
    "Millikan": millikan_distractor,
}


def main():
    rows = []
    for index, case in enumerate(cases()):
        log_space = case.name == "specific_heat_Cu"
        censored = case.name == "Onnes"
        fit, statistic, pvalue, valid = bootstrap_gof(
            case, DISTRACTORS[case.name], 1000,
            np.random.default_rng(20261900 + index), log_space, censored,
        )
        rows.append({
            "case": case.name,
            "distractor": fit.name,
            "statistic": statistic,
            "pvalue": pvalue,
            "bootstrap_valid": valid,
            "parameters": fit.fitted_parameters,
            "assumptions": fit.assumptions,
            "remains_rejected_5pct": bool(pvalue <= 0.05),
        })
        print(case.name, fit.name, pvalue)
    qvalues = bh_qvalues([row["pvalue"] for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["bh_qvalue"] = float(qvalue)
        row["remains_rejected_fdr5"] = bool(qvalue <= 0.05)
    result = {
        "status": "retrospective wrong-shape control; not a historical-theory benchmark",
        "question": "Does a prespecified low-dimensional distractor erase old-horizon mismatch?",
        "cases": rows,
    }
    out = ROOT / "results" / "irrelevant_horizon_controls.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
