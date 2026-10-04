"""Audit EPOCH real-case evidence at confirmatory operating points.

This script intentionally evaluates the statistical branch only.  It reports
finite-sample conformal-style tail probabilities against an independent,
size-matched classical calibration set, decisions at 5% and 1% FPR, and
Benjamini--Hochberg adjusted q-values across the four historical cases.

The historical 0.5 percentile cutoff is also shown, but is labelled as an
exploratory screening threshold rather than a rejection threshold.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def power(x, a, b, c):
    return a * np.abs(x) ** b + c


def exponential(x, a, b, c):
    return a * np.exp(np.clip(-b * x, -60, 60)) + c


FORMS = [
    (lambda x, a: a + 0 * x, [[1]]),
    (lambda x, a, b: a * x + b, [[1, 0]]),
    (power, [[1, 1, 0], [1, -1, 0], [1, 2, 0], [1, 0.5, 0], [1, -2, 0], [1, 1.5, 0]]),
    (exponential, [[1, 0.5, 0], [1, 2, 0], [1, 0.05, 0], [-1, 0.5, 0]]),
]


def fit_auto(x, y, train):
    best = np.inf
    fitted = None
    for form, starts in FORMS:
        for p0 in starts:
            try:
                params, _ = curve_fit(form, x[train], y[train], p0=p0, maxfev=6000)
                rmse = np.sqrt(np.mean((form(x[train], *params) - y[train]) ** 2))
                if np.isfinite(rmse) and rmse < best:
                    best = rmse
                    fitted = form, params
            except Exception:
                pass
    return fitted


def raw_score(x, y, combiner="legacy", return_details=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4 or np.std(y) < 1e-12:
        return None

    order = np.argsort(x)
    n = len(x)
    if n < 12:
        k = max(3, int(0.6 * n))
        splits = [(order[:k], order[k:]), (order[n - k :], order[: n - k])]
    else:
        splits = [(order[: int(0.35 * n)], order[int(0.7 * n) :]),
                  (order[int(0.65 * n) :], order[: int(0.3 * n)])]

    best = None
    for train, test in splits:
        if len(test) < 1:
            continue
        fitted = fit_auto(x, y, train)
        if fitted is None:
            continue
        form, params = fitted
        pred = form(x, *params)
        residual = np.abs(y - pred)
        floor = max(1e-3 * np.std(y), 0.01 * np.median(np.abs(y[train])))
        conformal_ratio = np.mean(residual[test] / (np.quantile(residual[train], 0.9) + floor + 1e-12))
        log_residual = np.abs(
            np.log(np.clip(np.abs(y) / np.clip(np.abs(pred), 1e-12, None), 1e-12, None))
        )
        boundary_ratio = np.nanmedian(log_residual[test]) / (max(np.nanmedian(log_residual[train]), 0.01) + 1e-6)
        r2 = 1 - np.sum((y[train] - pred[train]) ** 2) / (
            np.sum((y[train] - np.mean(y[train])) ** 2) + 1e-12
        )
        candidate = (r2, conformal_ratio, boundary_ratio)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None
    _, conformal_ratio, boundary_ratio = best
    conformal_scaled = conformal_ratio / 3.0
    boundary_scaled = boundary_ratio / 4.0
    if combiner == "legacy":
        # Reproduces the submitted implementation, including its ceiling at 3.
        score = 0.5 * (min(conformal_scaled, 3) + min(boundary_scaled, 3))
    elif combiner == "uncapped_log":
        # Diagnostic only until prospectively frozen: preserves ordering above
        # the legacy ceiling while limiting domination by a single component.
        score = 0.5 * (np.log1p(conformal_scaled) + np.log1p(boundary_scaled))
    else:
        raise ValueError(f"unknown combiner: {combiner}")
    if return_details:
        return {
            "score": float(score), "interior_r2": float(best[0]),
            "conformal_ratio": float(conformal_ratio), "boundary_ratio": float(boundary_ratio),
        }
    return score


def load_firas():
    rows = []
    for line in (DATA / "firas_monopole.txt").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or text[0].isalpha():
            continue
        fields = text.replace(",", " ").split()
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except (ValueError, IndexError):
            pass
    array = np.asarray(rows)
    return array[:, 0], array[:, 1]


def real_cases():
    nu, intensity = load_firas()
    cu = [-1.91844, -0.15973, 8.61013, -18.996, 21.9661, -12.7328, 3.54322, -0.3797, 0]
    temperature = np.logspace(np.log10(4), np.log10(300), 120)
    heat_capacity = 10 ** np.clip(
        sum(coef * np.log10(temperature) ** power for power, coef in enumerate(cu)), -30, 30
    )
    return {
        "FIRAS": (nu, intensity),
        "specific_heat_Cu": (temperature, heat_capacity),
        "Bertozzi": (np.array([0.5, 1, 1.5, 4.5, 15]), np.array([0.752, 0.828, 0.922, 0.974, 1.0])),
        "Onnes": (
            np.array([4.00, 4.10, 4.15, 4.19, 4.21, 4.25, 4.30, 4.35, 4.40]),
            np.array([1e-5, 1e-5, 1e-5, 1e-5, 0.110, 0.118, 0.126, 0.134, 0.142]),
        ),
    }


def upper_tail_pvalue(value, reference):
    """Finite-sample corrected upper-tail probability: (1 + # >=)/(n + 1)."""
    return float((1 + np.sum(reference >= value)) / (len(reference) + 1))


def bh_qvalues(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def build_calibration(n_cal, seed, combiner):
    clouds = np.load(DATA / "classical_pointclouds.npz", allow_pickle=True)["X"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(clouds), n_cal, replace=False)
    full, small = [], []
    for count, index in enumerate(indices, start=1):
        cloud = clouds[index]
        noisy_y = cloud[:, 1] * (1 + 0.01 * rng.standard_normal(len(cloud)))
        score = raw_score(cloud[:, 0], noisy_y, combiner=combiner)
        if score is not None:
            full.append(score)
        selection = rng.choice(len(cloud), 8, replace=False)
        score = raw_score(cloud[selection, 0], noisy_y[selection], combiner=combiner)
        if score is not None:
            small.append(score)
        if count % 50 == 0:
            print(f"calibration {count}/{n_cal}", flush=True)
    return np.asarray(full), np.asarray(small)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cal", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--combiner", choices=["legacy", "uncapped_log"], default="legacy")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "strict_evidence.json")
    args = parser.parse_args()

    full, small = build_calibration(args.n_cal, args.seed, args.combiner)
    rows = []
    for name, (x, y) in real_cases().items():
        value = raw_score(x, y, combiner=args.combiner)
        reference = small if len(x) < 20 else full
        pvalue = upper_tail_pvalue(value, reference)
        percentile = float(np.mean(reference < value))
        rows.append({
            "case": name,
            "n": len(x),
            "raw_score": value,
            "percentile_score": percentile,
            "tail_pvalue": pvalue,
            "screen_at_0.5": percentile > 0.5,
            "reject_at_5pct": pvalue <= 0.05,
            "reject_at_1pct": pvalue <= 0.01,
        })

    qvalues = bh_qvalues([row["tail_pvalue"] for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["bh_qvalue"] = float(qvalue)
        row["reject_fdr_5pct"] = bool(qvalue <= 0.05)

    result = {
        "calibration": {
            "requested": args.n_cal,
            "full_n": len(full),
            "small_n": len(small),
            "seed": args.seed,
            "combiner": args.combiner,
            "q95_full": float(np.quantile(full, 0.95)),
            "q99_full": float(np.quantile(full, 0.99)),
            "q95_small": float(np.quantile(small, 0.95)),
            "q99_small": float(np.quantile(small, 0.99)),
        },
        "cases": rows,
        "interpretation": (
            "The 0.5 percentile rule is exploratory screening. Confirmatory decisions use the "
            "finite-sample upper-tail p-value and a pre-specified alpha/FDR rule."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
