"""Paired old-horizon -> advanced-horizon experiment for unified EPOCH.

Each case uses the identical x-grid under both horizons.  The theory-conditioned
component fits only declared nuisance parameters, evaluates a standardized
lack-of-fit statistic, and obtains an empirical p-value from a parametric
counterfactual null on the same grid and with the same error model.

This is a development experiment.  Error floors for digitized/reference-curve
data are explicit sensitivity assumptions, not published raw covariance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit, least_squares


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def bh_qvalues(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def load_firas():
    rows = []
    for line in (DATA / "firas_monopole.txt").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        try:
            rows.append(tuple(float(value) for value in fields[:5]))
        except (ValueError, IndexError):
            pass
    values = np.asarray(rows)
    x, y = values[:, 0], values[:, 1]
    published_sigma = values[:, 3] / 1000.0
    sigma = np.sqrt(published_sigma**2 + (0.001 * y) ** 2)
    return x, y, sigma


def cp_reference():
    row = json.loads((DATA / "nist_cp_coef.json").read_text(encoding="utf-8"))[0]
    coefficients = row[1]
    lo, hi = row[2]
    temperature = np.logspace(np.log10(lo), np.log10(hi), 120)
    log_t = np.log10(temperature)
    cp = 10 ** sum(coef * log_t**power for power, coef in enumerate(coefficients))
    return temperature, cp


def make_debye_interpolator():
    z_grid = np.geomspace(1e-3, 120, 900)
    values = []
    for z in z_grid:
        integral = quad(
            lambda u: u**4 * np.exp(-u) / max((1 - np.exp(-u)) ** 2, 1e-300),
            0, z, epsabs=1e-9, epsrel=1e-8, limit=200,
        )[0]
        values.append(3 * integral / z**3)
    interpolation = PchipInterpolator(np.log(z_grid), np.asarray(values), extrapolate=True)

    def normalized_cv(z):
        z = np.clip(np.asarray(z, float), z_grid[0], z_grid[-1])
        return np.clip(interpolation(np.log(z)), 1e-12, 1.05)

    return normalized_cv


DEBYE_CV = make_debye_interpolator()


@dataclass
class HorizonFit:
    name: str
    prediction: np.ndarray
    refit: callable
    fitted_parameters: dict
    assumptions: str


@dataclass
class ClockCase:
    name: str
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    old_fit: callable
    new_fit: callable
    provenance: str


def fit_firas_old(x, y, sigma):
    weight = 1 / sigma**2
    amplitude = np.sum(weight * y * x**2) / np.sum(weight * x**4)

    def refit(y_new):
        return fit_firas_old(x, y_new, sigma)

    return HorizonFit("Rayleigh-Jeans shape", amplitude * x**2, refit, {"amplitude": amplitude},
                      "Published pointwise sigma plus a 0.1% independent systematic floor; covariance unavailable locally.")


def fit_firas_new(x, y, sigma):
    def model(xv, log_a, b):
        return np.exp(log_a) * xv**3 / np.expm1(np.clip(b * xv, 1e-8, 60))

    params, _ = curve_fit(model, x, y, p0=[np.log(20), 0.5], sigma=sigma, absolute_sigma=True,
                          bounds=([-20, 1e-4], [20, 10]), maxfev=20000)

    def refit(y_new):
        return fit_firas_new(x, y_new, sigma)

    return HorizonFit("Planck spectral shape", model(x, *params), refit,
                      {"log_amplitude": float(params[0]), "inverse_temperature_scale": float(params[1])},
                      "Successor functional form fixed; amplitude and temperature scale are nuisance fits.")


def fit_bertozzi_old(x, y, sigma):
    pred = 2 * x / 0.511
    return HorizonFit("Newtonian kinetic energy", pred, lambda y_new: fit_bertozzi_old(x, y_new, sigma), {},
                      "Electron rest energy fixed at 0.511 MeV; 0.02 beta^2 digitization/error floor.")


def fit_bertozzi_new(x, y, sigma):
    pred = 1 - 1 / (1 + x / 0.511) ** 2
    return HorizonFit("Relativistic kinetic energy", pred, lambda y_new: fit_bertozzi_new(x, y_new, sigma), {},
                      "No fitted physical parameter; electron rest energy fixed at 0.511 MeV.")


def fit_cp_old(x, y, sigma):
    interior = x >= np.quantile(x, 0.75)
    log_y = np.log(y)
    level = np.average(log_y[interior], weights=1 / sigma[interior] ** 2)
    pred = np.exp(np.full_like(x, level))

    def refit(y_new):
        return fit_cp_old(x, y_new, sigma)

    return HorizonFit("Dulong-Petit constant", pred, refit, {"log_plateau": float(level)},
                      "Plateau nuisance fitted only on the upper temperature quartile.")


def fit_cp_new(x, y, sigma):
    log_y = np.log(y)

    def log_model(params):
        plateau, theta, gamma = np.exp(params)
        pred = plateau * DEBYE_CV(theta / x) + gamma * x
        return np.log(np.clip(pred, 1e-30, None))

    result = least_squares(lambda p: (log_y - log_model(p)) / sigma,
                           x0=np.log([390, 340, 1e-3]),
                           bounds=(np.log([10, 10, 1e-8]), np.log([5000, 5000, 10])), max_nfev=3000)
    plateau, theta, gamma = np.exp(result.x)
    pred = np.exp(log_model(result.x))

    def refit(y_new):
        return fit_cp_new(x, y_new, sigma)

    return HorizonFit("Debye plus electronic term", pred, refit,
                      {"plateau": float(plateau), "theta_K": float(theta), "gamma": float(gamma)},
                      "NIST reference curve treated with a 5% relative error floor; not a raw-measurement covariance.")


def fit_onnes_old(x, y, sigma):
    normal = x >= 4.21
    params = np.polyfit(x[normal], y[normal], 1, w=1 / sigma[normal])
    pred = np.polyval(params, x)
    pred = np.clip(pred, 1e-8, None)

    def refit(y_new):
        return fit_onnes_old(x, y_new, sigma)

    return HorizonFit("smooth normal-state resistance", pred, refit,
                      {"slope": float(params[0]), "intercept": float(params[1])},
                      "Normal trend fitted above 4.21 K; sub-transition values are upper limits represented at the limit.")


def fit_onnes_new(x, y, sigma):
    normal = x >= 4.21
    params = np.polyfit(x[normal], y[normal], 1, w=1 / sigma[normal])
    pred = np.where(x < 4.20, 1e-5, np.polyval(params, x))
    pred = np.clip(pred, 1e-8, None)

    def refit(y_new):
        return fit_onnes_new(x, y_new, sigma)

    return HorizonFit("superconducting transition", pred, refit,
                      {"Tc_K": 4.20, "normal_slope": float(params[0]), "normal_intercept": float(params[1])},
                      "Tc and censoring limit fixed from the historical record; normal trend is a nuisance fit.")


def fit_millikan_old(x, y, sigma):
    level = np.average(y, weights=1 / sigma**2)
    pred = np.full_like(y, level)
    return HorizonFit("frequency-independent energy", pred, lambda y_new: fit_millikan_old(x, y_new, sigma),
                      {"constant_V": float(level)}, "0.02 V pointwise error from the case-study protocol.")


def fit_millikan_new(x, y, sigma):
    params = np.polyfit(x, y, 1, w=1 / sigma)
    pred = np.polyval(params, x)
    return HorizonFit("linear photoelectric relation", pred, lambda y_new: fit_millikan_new(x, y_new, sigma),
                      {"slope_V_per_1e14Hz": float(params[0]), "intercept_V": float(params[1])},
                      "Successor shape fixed to a line; slope and work-function intercept are nuisance fits.")


def cases():
    fx, fy, fs = load_firas()
    cp_x, cp_y = cp_reference()
    return [
        ClockCase("FIRAS", fx, fy, fs, fit_firas_old, fit_firas_new, "COBE-FIRAS table with published pointwise errors"),
        ClockCase("Bertozzi", np.array([0.5, 1, 1.5, 4.5, 15.0]),
                  np.array([0.752, 0.828, 0.922, 0.974, 1.0]), np.full(5, 0.02),
                  fit_bertozzi_old, fit_bertozzi_new, "five digitized values; conservative beta^2 error floor"),
        ClockCase("specific_heat_Cu", cp_x, cp_y, np.full(len(cp_x), 0.05),
                  fit_cp_old, fit_cp_new, "NIST reference polynomial; analysed in log space"),
        ClockCase("Onnes", np.array([4.00, 4.10, 4.15, 4.19, 4.21, 4.25, 4.30, 4.35, 4.40]),
                  np.array([1e-5, 1e-5, 1e-5, 1e-5, 0.110, 0.118, 0.126, 0.134, 0.142]),
                  np.array([2e-6] * 4 + [0.003] * 5), fit_onnes_old, fit_onnes_new,
                  "digitized values; first four observations are censoring limits"),
        ClockCase("Millikan", np.array([5.49, 7.41, 8.21, 9.59]), np.array([0.45, 1.25, 1.58, 2.14]),
                  np.full(4, 0.02), fit_millikan_old, fit_millikan_new,
                  "four sodium stopping-potential values"),
    ]


def residual_statistic(y, pred, sigma, log_space=False):
    if log_space:
        return float(np.sum(((np.log(np.clip(y, 1e-30, None)) - np.log(np.clip(pred, 1e-30, None))) / sigma) ** 2))
    return float(np.sum(((y - pred) / sigma) ** 2))


def bootstrap_gof(case, fit_function, iterations, rng, log_space=False, censored=False):
    fitted = fit_function(case.x, case.y, case.sigma)
    observed = residual_statistic(case.y, fitted.prediction, case.sigma, log_space=log_space)
    simulated = []
    for _ in range(iterations):
        if log_space:
            y_sim = fitted.prediction * np.exp(case.sigma * rng.standard_normal(len(case.y)))
        else:
            y_sim = fitted.prediction + case.sigma * rng.standard_normal(len(case.y))
        if censored:
            y_sim = np.maximum(y_sim, 1e-5)
        try:
            refitted = fit_function(case.x, y_sim, case.sigma)
            simulated.append(residual_statistic(y_sim, refitted.prediction, case.sigma, log_space=log_space))
        except Exception:
            continue
    simulated = np.asarray(simulated)
    pvalue = float((1 + np.sum(simulated >= observed)) / (len(simulated) + 1))
    return fitted, observed, pvalue, len(simulated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "theory_clock_experiment.json")
    args = parser.parse_args()
    rows = []
    for index, case in enumerate(cases()):
        rng_old = np.random.default_rng(args.seed + 2 * index)
        rng_new = np.random.default_rng(args.seed + 2 * index + 1)
        log_space = case.name == "specific_heat_Cu"
        censored = case.name == "Onnes"
        old, old_stat, old_p, old_n = bootstrap_gof(case, case.old_fit, args.bootstrap, rng_old, log_space, censored)
        new, new_stat, new_p, new_n = bootstrap_gof(case, case.new_fit, args.bootstrap, rng_new, log_space, censored)
        row = {
            "case": case.name, "n": len(case.y), "provenance": case.provenance,
            "old_horizon": {"model": old.name, "statistic": old_stat, "pvalue": old_p,
                            "bootstrap_valid": old_n, "parameters": old.fitted_parameters,
                            "assumptions": old.assumptions},
            "advanced_horizon": {"model": new.name, "statistic": new_stat, "pvalue": new_p,
                                 "bootstrap_valid": new_n, "parameters": new.fitted_parameters,
                                 "assumptions": new.assumptions},
            "score_collapse_ratio": float(old_stat / max(new_stat, 1e-12)),
            "paired_success": bool(old_p <= 0.05 and new_p > 0.05),
        }
        rows.append(row)
        print(case.name, "old p", old_p, "new p", new_p, "collapse", row["score_collapse_ratio"])

    old_q = bh_qvalues([row["old_horizon"]["pvalue"] for row in rows])
    new_q = bh_qvalues([row["advanced_horizon"]["pvalue"] for row in rows])
    for row, oq, nq in zip(rows, old_q, new_q):
        row["old_horizon"]["bh_qvalue"] = float(oq)
        row["advanced_horizon"]["bh_qvalue"] = float(nq)
        row["paired_success_fdr5"] = bool(oq <= 0.05 and nq > 0.05)

    result = {
        "status": "development experiment; digitized/reference-data error models require sensitivity analysis",
        "test": "same-grid parametric-bootstrap goodness of fit under old and advanced horizons",
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
