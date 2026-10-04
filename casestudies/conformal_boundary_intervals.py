"""Split-conformal calibration of boundary intervals over synthetic mechanisms."""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from boundary_interval_coverage import FAMILIES, ROOT, localize, truth, wilson


def simulate(task):
    family, n, noise, repeat = task
    seed = 90_000_000 * FAMILIES.index(family) + 100_000 * n + int(noise * 1000) * 1000 + repeat
    rng = np.random.default_rng(seed)
    x0 = 3.0
    x = np.sort(rng.uniform(0.15, 6.0, n))
    y_true = truth(family, x, x0, 0.35)
    sigma = np.full(n, noise * max(np.std(y_true), 1e-8))
    y = y_true + rng.normal(0, sigma)
    estimate = localize(x, y, sigma)
    return {
        "family": family, "n": n, "noise": noise, "repeat": repeat, "estimate": estimate,
        "true": x0, "range": float(np.ptp(x)),
        "error_fraction": None if estimate is None else float(abs(estimate - x0) / np.ptp(x)),
    }


def main():
    repeats = 40
    tasks = [(family, n, noise, repeat) for family in FAMILIES for n in [40, 80, 160]
             for noise in [0.01, 0.05, 0.10] for repeat in range(repeats)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(simulate, tasks, chunksize=4))
    calibration = [row for row in rows if row["repeat"] < repeats // 2 and row["error_fraction"] is not None]
    test = [row for row in rows if row["repeat"] >= repeats // 2]
    radius_by_n = {}
    for sample_size in [40, 80, 160]:
        errors = np.sort([row["error_fraction"] for row in calibration if row["n"] == sample_size])
        rank = min(math.ceil((len(errors) + 1) * 0.90) - 1, len(errors) - 1)
        radius_by_n[sample_size] = float(errors[rank])
    for row in test:
        if row["estimate"] is None:
            row["abstain"] = True
            row["covered"] = False
            continue
        radius = radius_by_n[row["n"]] * row["range"]
        row["interval90"] = [row["estimate"] - radius, row["estimate"] + radius]
        row["covered"] = bool(row["interval90"][0] <= row["true"] <= row["interval90"][1])
        row["abstain"] = False
    retained = [row for row in test if not row["abstain"]]

    def summary(block):
        usable = [row for row in block if not row["abstain"]]
        hits = sum(row["covered"] for row in usable)
        return {
            "n": len(block), "retained": len(usable), "abstention_rate": 1 - len(usable) / len(block),
            "coverage": hits / len(usable), "coverage_wilson95": wilson(hits, len(usable)),
        }

    result = {
        "status": "held-out simulation-calibrated boundary interval",
        "calibration_datasets": len(calibration), "test_datasets": len(test),
        "nominal_coverage": 0.90,
        "radius_fraction_of_x_range_by_sample_size": {str(key): value for key, value in radius_by_n.items()},
        "overall": summary(test),
        "by_family": {family: summary([row for row in test if row["family"] == family]) for family in FAMILIES},
        "by_noise": {str(noise): summary([row for row in test if row["noise"] == noise]) for noise in [0.01, 0.05, 0.10]},
        "by_sample_size": {str(n): summary([row for row in test if row["n"] == n]) for n in [40, 80, 160]},
        "limitation": "Coverage is marginal over the declared synthetic mechanism distribution, not guaranteed for a new mechanism.",
    }
    path = ROOT / "results" / "conformal_boundary_intervals.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
