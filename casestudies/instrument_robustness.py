"""Stress-test EPOCH's data-anchored component on instrument-induced hard negatives."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from calibrated_detector_v2 import CLASSICAL_FAMILIES
from strict_evidence_audit import raw_score


def corrupt(x, y, kind, severity, rng):
    x = x.copy()
    y = y.copy()
    span = np.ptp(x)
    scale = max(np.std(y), 1e-9)
    u = (x - x.min()) / max(span, 1e-12)
    if kind == "clean":
        pass
    elif kind == "calibration_drift":
        y *= 1 + severity * 0.20 * (u - 0.5)
    elif kind == "sensor_saturation":
        cap = np.quantile(y, max(0.55, 1 - 0.40 * severity))
        y = np.minimum(y, cap)
    elif kind == "censoring":
        floor = np.quantile(y, min(0.45, 0.45 * severity))
        y = np.maximum(y, floor)
    elif kind == "x_axis_error":
        x += rng.normal(0, severity * 0.08 * span, len(x))
    elif kind == "outliers":
        count = max(1, int(np.ceil(severity * 0.08 * len(y))))
        indices = rng.choice(len(y), count, replace=False)
        y[indices] += rng.normal(0, 6 * scale, count)
    elif kind == "correlated_noise":
        rho = 0.95 * severity
        innovation = rng.normal(0, 0.05 * scale, len(y))
        noise = np.zeros(len(y))
        for index in range(1, len(y)):
            noise[index] = rho * noise[index - 1] + innovation[index]
        y += noise
    elif kind == "heteroscedasticity":
        y += rng.normal(0, 0.02 * scale * (1 + 8 * severity * u), len(y))
    else:
        raise KeyError(kind)
    return x, y


def task_run(task):
    family, kind, severity, repeat, n = task
    seed = 100_000 * repeat + 1000 * list(CLASSICAL_FAMILIES).index(family) + int(severity * 100)
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.15, 6.0, n))
    y = CLASSICAL_FAMILIES[family](x)
    y += rng.normal(0, 0.02 * max(np.std(y), 1e-9), n)
    x, y = corrupt(x, y, kind, severity, rng)
    return family, kind, severity, raw_score(x, y, combiner="uncapped_log")


def wilson_interval(successes, total, z=1.96):
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main():
    detector = json.loads((ROOT / "results" / "calibrated_detector_v2.json").read_text(encoding="utf-8"))
    threshold_5 = detector["calibration"]["120"]["q95"]
    threshold_1 = detector["calibration"]["120"]["q99"]
    kinds = ["clean", "calibration_drift", "sensor_saturation", "censoring", "x_axis_error",
             "outliers", "correlated_noise", "heteroscedasticity"]
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    tasks = [
        (family, kind, severity, repeat, 120)
        for family in CLASSICAL_FAMILIES
        for kind in kinds
        for severity in severities
        for repeat in range(10)
        if kind != "clean" or severity == 0.0
    ]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(task_run, tasks, chunksize=4))

    summary = []
    for kind in kinds:
        levels = [0.0] if kind == "clean" else severities
        for severity in levels:
            block = [score for family, row_kind, row_severity, score in rows
                     if row_kind == kind and row_severity == severity and np.isfinite(score)]
            fp5 = int(np.sum(np.asarray(block) > threshold_5))
            fp1 = int(np.sum(np.asarray(block) > threshold_1))
            summary.append({
                "corruption": kind, "severity": severity, "n": len(block),
                "fpr_at_nominal_5pct": fp5 / len(block), "fpr5_wilson_95ci": wilson_interval(fp5, len(block)),
                "fpr_at_nominal_1pct": fp1 / len(block), "fpr1_wilson_95ci": wilson_interval(fp1, len(block)),
            })
            print(kind, severity, "FPR5", fp5 / len(block), "FPR1", fp1 / len(block))

    output = {
        "status": "synthetic instrument hard-negative stress test",
        "thresholds": {"5pct": threshold_5, "1pct": threshold_1},
        "severity_definition": {
            "calibration_drift": "up to 20% end-to-end multiplicative drift",
            "sensor_saturation": "clip down to the 60th response percentile",
            "censoring": "raise values below up to the 45th response percentile to a floor",
            "x_axis_error": "up to 8% of x-range Gaussian error",
            "outliers": "up to 8% of points displaced by six response standard deviations",
            "correlated_noise": "AR(1) coefficient up to 0.95",
            "heteroscedasticity": "noise scale grows up to nine-fold across x",
        },
        "summary": summary,
    }
    path = ROOT / "results" / "instrument_robustness.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
