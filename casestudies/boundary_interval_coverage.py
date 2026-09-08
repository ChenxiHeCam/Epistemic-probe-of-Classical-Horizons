"""Synthetic coverage audit for EPOCH boundary intervals."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from strict_evidence_audit import fit_auto


FAMILIES = ["step", "saturation", "kink", "smooth_turnover", "smooth_crossover"]


def truth(family, x, x0, width):
    if family == "step":
        return x + np.where(x >= x0, 2.0, 0.0)
    if family == "saturation":
        return np.where(x < x0, x**2, x0**2)
    if family == "kink":
        return np.where(x < x0, x, x0 + 0.2 * (x - x0))
    if family == "smooth_turnover":
        return x**2 / (1 + (np.maximum(x - x0, 0) / width) ** 2)
    if family == "smooth_crossover":
        blend = 1 / (1 + np.exp(-(x - x0) / width))
        successor = x0 + 0.15 * (x - x0)
        return (1 - blend) * x + blend * successor
    raise KeyError(family)


def fit_incumbent(x, y):
    order = np.argsort(x)
    train = order[: max(8, int(0.30 * len(x)))]
    return fit_auto(x, y, train), train


def localize(x, y, sigma):
    fitted, train = fit_incumbent(x, y)
    if fitted is None:
        return None
    form, params = fitted
    order = np.argsort(x)
    pred = form(x, *params)
    floor = np.maximum(sigma, 0.01 * np.median(np.abs(y[train])))
    standardized = np.abs(y - pred) / floor
    values = standardized[order]
    start = len(train)
    for index in range(start, len(order) - 3):
        if np.sum(values[index : index + 4] > 3.0) >= 3:
            return float(x[order[index]])
    return None


def one_dataset(task):
    family, n, noise_fraction, repeat, boot = task
    seed = 10_000_000 * FAMILIES.index(family) + 100_000 * n + int(noise_fraction * 1000) * 100 + repeat
    rng = np.random.default_rng(seed)
    x0 = 3.0
    width = 0.35
    x = np.sort(rng.uniform(0.15, 6.0, n))
    y_true = truth(family, x, x0, width)
    sigma = np.full(n, noise_fraction * max(np.std(y_true), 1e-8))
    y = y_true + rng.normal(0, sigma)
    estimate = localize(x, y, sigma)
    draws = []
    for _ in range(boot):
        y_boot = y + rng.normal(0, sigma)
        location = localize(x, y_boot, sigma)
        if location is not None:
            draws.append(location)
    valid_fraction = len(draws) / boot
    if estimate is None or valid_fraction < 0.80:
        return {
            "family": family, "n": n, "noise": noise_fraction, "repeat": repeat,
            "estimate": estimate, "valid_bootstrap_fraction": valid_fraction, "abstain": True,
        }
    low, high = np.quantile(draws, [0.05, 0.95])
    width_fraction = (high - low) / (x.max() - x.min())
    abstain = width_fraction > 0.50
    return {
        "family": family, "n": n, "noise": noise_fraction, "repeat": repeat,
        "estimate": estimate, "true_boundary": x0, "interval90": [float(low), float(high)],
        "valid_bootstrap_fraction": valid_fraction, "interval_width_fraction": float(width_fraction),
        "absolute_error_fraction": float(abs(estimate - x0) / (x.max() - x.min())),
        "covered": bool(low <= x0 <= high), "abstain": bool(abstain),
    }


def wilson(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    p = successes / total
    d = 1 + z**2 / total
    c = (p + z**2 / (2 * total)) / d
    r = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    return [max(0, c - r), min(1, c + r)]


def summarize(block):
    retained = [row for row in block if not row["abstain"]]
    covered = sum(row.get("covered", False) for row in retained)
    return {
        "datasets": len(block), "retained": len(retained), "abstention_rate": 1 - len(retained) / len(block),
        "coverage90": covered / len(retained) if retained else None,
        "coverage90_wilson95": wilson(covered, len(retained)),
        "median_interval_width_fraction": float(np.median([row["interval_width_fraction"] for row in retained])) if retained else None,
        "median_absolute_error_fraction": float(np.median([row["absolute_error_fraction"] for row in retained])) if retained else None,
    }


def main():
    tasks = [
        (family, n, noise, repeat, 100)
        for family in FAMILIES for n in [40, 80, 160] for noise in [0.01, 0.05, 0.10] for repeat in range(8)
    ]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one_dataset, tasks, chunksize=1))
    by_family = {family: summarize([row for row in rows if row["family"] == family]) for family in FAMILIES}
    by_noise = {str(noise): summarize([row for row in rows if row["noise"] == noise]) for noise in [0.01, 0.05, 0.10]}
    by_n = {str(n): summarize([row for row in rows if row["n"] == n]) for n in [40, 80, 160]}
    result = {
        "status": "development coverage audit",
        "interval": "90% measurement-bootstrap interval; 100 perturbations per dataset",
        "abstention": "fewer than 80% valid bootstrap locations or interval wider than 50% of x-range",
        "overall": summarize(rows), "by_family": by_family, "by_noise": by_noise, "by_sample_size": by_n,
        "rows": rows,
    }
    path = ROOT / "results" / "boundary_interval_coverage.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
