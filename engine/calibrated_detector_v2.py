"""Prospective candidate for EPOCH's shared calibrated decision layer.

This is a development experiment, not pre-registered evidence.  It removes the
legacy score ceiling, constructs calibration distributions matched to the
actual sample counts of the historical cases, and reports TPR at 5% and 1% FPR
with mechanism-clustered bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from strict_evidence_audit import DATA, ROOT, bh_qvalues, real_cases, raw_score, upper_tail_pvalue


def _calibration_task(task):
    cloud, sample_size, seed = task
    rng = np.random.default_rng(seed)
    replace = sample_size > len(cloud)
    select = rng.choice(len(cloud), sample_size, replace=replace)
    x = cloud[select, 0]
    y = cloud[select, 1]
    y = y * (1 + 0.01 * rng.standard_normal(sample_size))
    return raw_score(x, y, combiner="uncapped_log")


CLASSICAL_FAMILIES = {
    "linear": lambda x: 2 * x + 1,
    "quadratic": lambda x: 0.5 * x**2 + 0.2,
    "power_1p5": lambda x: x**1.5,
    "inverse": lambda x: 3 / (x + 0.2),
    "inverse_square": lambda x: 2 / (x + 0.2) ** 2,
    "exponential_decay": lambda x: 2 * np.exp(-0.7 * x) + 0.1,
    "exponential_growth": lambda x: 0.2 * np.exp(0.5 * x),
    "square_root": lambda x: 2 * np.sqrt(x),
    "power_2p5": lambda x: 0.3 * x**2.5,
    "affine_decline": lambda x: 8 - x,
    "quartic": lambda x: 0.02 * x**4,
    "offset_inverse": lambda x: 1 / (1 + x),
}


def _breakdown(name, x):
    if name == "saturation":
        return np.minimum(x**2, 6.0)
    if name == "step":
        return x + np.where(x > 3.0, 4.0, 0.0)
    if name == "kink":
        return np.where(x <= 3.0, x, 3.0 + 0.2 * (x - 3.0))
    if name == "smooth_rolloff":
        return x**2 / (1 + np.maximum(x - 2.5, 0) ** 2)
    if name == "smooth_crossover":
        blend = 1 / (1 + np.exp(-(x - 3.0) / 0.25))
        return (1 - blend) * x + blend * (8 - 0.5 * x)
    if name == "threshold":
        return np.maximum(x - 2.5, 0) + 0.02
    if name == "turnover":
        return x**2 * np.exp(-0.5 * x)
    if name == "oscillation":
        return 2 + np.sin(4 * x)
    if name == "resonance":
        return 0.2 * x + 2 / (1 + ((x - 3) / 0.2) ** 2)
    if name == "discontinuous_drop":
        return np.where(x < 3.0, x + 1, 0.02)
    if name == "log_correction":
        return x**1.5 * (1 + 0.8 * np.log1p(x))
    if name == "bounded_growth":
        return 1 - np.exp(-1.5 * x)
    raise KeyError(name)


BREAKDOWN_FAMILIES = [
    "saturation", "step", "kink", "smooth_rolloff", "smooth_crossover", "threshold",
    "turnover", "oscillation", "resonance", "discontinuous_drop", "log_correction", "bounded_growth",
]


def _benchmark_task(task):
    label, family, seed, sample_size = task
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.15, 6.0, sample_size))
    y = CLASSICAL_FAMILIES[family](x) if label == 0 else _breakdown(family, x)
    scale = max(np.std(y), 1e-8)
    y = y + rng.normal(0, 0.02 * scale, sample_size)
    return label, family, seed, raw_score(x, y, combiner="uncapped_log")


def percentile_interval(values, rng, iterations=5000):
    values = np.asarray(values, float)
    draws = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(iterations)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cal", type=int, default=800)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--benchmark-repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "calibrated_detector_v2.json")
    args = parser.parse_args()

    clouds = np.load(DATA / "classical_pointclouds.npz", allow_pickle=True)["X"]
    cases = real_cases()
    sample_sizes = sorted({len(x) for x, _ in cases.values()})
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(clouds), args.n_cal, replace=False)

    tasks = []
    for size in sample_sizes:
        for position, index in enumerate(indices):
            tasks.append((clouds[index], size, args.seed + size * 100_000 + position))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        values = list(pool.map(_calibration_task, tasks, chunksize=4))

    calibration = {}
    offset = 0
    for size in sample_sizes:
        block = np.asarray(values[offset : offset + args.n_cal], float)
        block = block[np.isfinite(block)]
        calibration[size] = block
        offset += args.n_cal
        print(f"calibration n={size}: valid={len(block)} q95={np.quantile(block, .95):.4g} q99={np.quantile(block, .99):.4g}")

    case_rows = []
    for name, (x, y) in cases.items():
        score = raw_score(x, y, combiner="uncapped_log")
        reference = calibration[len(x)]
        pvalue = upper_tail_pvalue(score, reference)
        case_rows.append({
            "case": name, "n": len(x), "score": score, "pvalue": pvalue,
            "reject_5pct": pvalue <= 0.05, "reject_1pct": pvalue <= 0.01,
        })
    qvalues = bh_qvalues([row["pvalue"] for row in case_rows])
    for row, qvalue in zip(case_rows, qvalues):
        row["bh_qvalue"] = float(qvalue)
        row["reject_bh_5pct"] = bool(qvalue <= 0.05)

    benchmark_n = 120
    benchmark_tasks = []
    for repeat in range(args.benchmark_repeats):
        for family in CLASSICAL_FAMILIES:
            benchmark_tasks.append((0, family, args.seed + repeat * 100 + len(benchmark_tasks), benchmark_n))
        for family in BREAKDOWN_FAMILIES:
            benchmark_tasks.append((1, family, args.seed + repeat * 100 + len(benchmark_tasks), benchmark_n))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        benchmark = list(pool.map(_benchmark_task, benchmark_tasks, chunksize=2))

    ref = calibration[benchmark_n]
    thresholds = {"5pct": float(np.quantile(ref, 0.95)), "1pct": float(np.quantile(ref, 0.99))}
    operating_points = {}
    for alpha_name, threshold in thresholds.items():
        positive_by_family = {}
        negative_by_family = {}
        for label, family, seed, score in benchmark:
            if score is None or not np.isfinite(score):
                continue
            target = positive_by_family if label == 1 else negative_by_family
            target.setdefault(family, []).append(float(score > threshold))
        positive_means = {family: float(np.mean(hit)) for family, hit in positive_by_family.items()}
        negative_means = {family: float(np.mean(hit)) for family, hit in negative_by_family.items()}
        positive_values = list(positive_means.values())
        negative_values = list(negative_means.values())
        operating_points[alpha_name] = {
            "tpr": float(np.mean(positive_values)),
            "tpr_mechanism_bootstrap_95ci": percentile_interval(
                positive_values, np.random.default_rng(args.seed + len(alpha_name))
            ),
            "fpr": float(np.mean(negative_values)),
            "fpr_family_bootstrap_95ci": percentile_interval(
                negative_values, np.random.default_rng(args.seed + 100 + len(alpha_name))
            ),
            "breakdown_by_family": positive_means,
            "classical_by_family": negative_means,
        }

    fit_failures = {}
    for label, label_name in [(0, "classical"), (1, "breakdown")]:
        rows = [score for row_label, _, _, score in benchmark if row_label == label]
        failed = sum(score is None or not np.isfinite(score) for score in rows)
        fit_failures[label_name] = {
            "n": len(rows),
            "failed_or_abstained": failed,
            "rate": float(failed / len(rows)),
        }

    result = {
        "status": "development candidate; must be frozen before sealed evaluation",
        "task_object": {
            "admissible_models": ["constant", "affine", "power", "exponential"],
            "fitted_parameters": "all parameters of each candidate form; multiple fixed initializations",
            "observation_model": "source point cloud resampled to the target n with independent 1% multiplicative Gaussian noise",
            "fitting_data": "both predeclared endpoint interiors are tried; 60% at n<12, otherwise 35%",
            "adaptive_procedure": [
                "sort by x",
                "try low-to-high and high-to-low endpoint directions",
                "fit every admissible form and initialization on each candidate interior",
                "retain the direction whose selected form has the larger interior R2",
                "compute conformal-residual and log-extrapolation ratios",
                "combine without clipping",
            ],
            "calibration_scope": "the complete direction, model, parameter-fit and scoring procedure is rerun for every calibration and test dataset",
            "failure_criterion": "score strictly above the size-matched q95 primary or q99 secondary calibration threshold",
            "semantic_scope": "incompatibility with this four-form extrapolation task, not rejection of classical physics in general",
        },
        "score": "0.5*(log1p(conformal_ratio/3)+log1p(boundary_ratio/4))",
        "calibration": {
            str(size): {
                "valid_n": len(reference),
                "q95": float(np.quantile(reference, 0.95)),
                "q99": float(np.quantile(reference, 0.99)),
            }
            for size, reference in calibration.items()
        },
        "real_cases": case_rows,
        "synthetic_mechanism_benchmark": {
            "sample_size": benchmark_n,
            "repeats_per_family": args.benchmark_repeats,
            "classical_families": list(CLASSICAL_FAMILIES),
            "breakdown_families": BREAKDOWN_FAMILIES,
            "thresholds": thresholds,
            "operating_points": operating_points,
            "fit_failure_or_abstention": fit_failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
