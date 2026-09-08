"""Paired epistemic-clock test for the two frozen EPOCH horizon models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_test(decreases: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    tail = min(decreases, non_ties - decreases)
    probability = sum(math.comb(non_ties, k) for k in range(tail + 1)) / 2**non_ties
    return float(min(1.0, 2.0 * probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre1900", type=Path, required=True)
    parser.add_argument("--pre1950", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    pre1900_path = args.pre1900.resolve()
    pre1950_path = args.pre1950.resolve()
    old = json.loads(pre1900_path.read_text(encoding="utf-8"))
    new = json.loads(pre1950_path.read_text(encoding="utf-8"))
    if old["knowledge_cutoff"] != 1899 or new["knowledge_cutoff"] != 1950:
        raise RuntimeError("clock comparison requires 1899 and 1950 frozen evaluations")
    if old["primary_metric"] != new["primary_metric"]:
        raise RuntimeError("primary score mismatch")
    if (
        old["future_horizon_used_for_training_or_score_fitting"]
        or new["future_horizon_used_for_training_or_score_fitting"]
    ):
        raise RuntimeError("a horizon evaluation reports future-data fitting")

    metric = old["primary_metric"]
    old_rows = old["sets"]["curated_1901_1950"]["scores"]
    new_rows = new["sets"]["curated_1901_1950"]["scores"]
    families = sorted(set(old_rows) & set(new_rows))
    if set(old_rows) != set(new_rows):
        raise RuntimeError("paired clock family sets do not match")
    paired = {
        family: {
            "pre1900_score": float(old_rows[family][metric]),
            "pre1950_score": float(new_rows[family][metric]),
            "collapse": float(old_rows[family][metric] - new_rows[family][metric]),
        }
        for family in families
    }
    collapse = np.asarray([paired[family]["collapse"] for family in families])
    decreases = int((collapse > 0).sum())
    increases = int((collapse < 0).sum())
    ties = int((collapse == 0).sum())
    rng = np.random.default_rng(args.seed)
    medians = np.empty(args.bootstrap, dtype=np.float64)
    means = np.empty(args.bootstrap, dtype=np.float64)
    for index in range(args.bootstrap):
        sample = collapse[rng.integers(0, len(collapse), len(collapse))]
        medians[index] = np.median(sample)
        means[index] = np.mean(sample)
    result = {
        "status": "PASS",
        "schema_version": "3.0",
        "test": "paired anomaly-score collapse after advancing knowledge horizon",
        "score": metric,
        "families": len(families),
        "pre1900_median": float(np.median([
            paired[family]["pre1900_score"] for family in families
        ])),
        "pre1950_median": float(np.median([
            paired[family]["pre1950_score"] for family in families
        ])),
        "median_collapse": float(np.median(collapse)),
        "median_collapse_ci95": [
            float(value) for value in np.quantile(medians, [0.025, 0.975])
        ],
        "mean_collapse": float(np.mean(collapse)),
        "mean_collapse_ci95": [
            float(value) for value in np.quantile(means, [0.025, 0.975])
        ],
        "families_decreased": decreases,
        "families_increased": increases,
        "ties": ties,
        "fraction_decreased_among_non_ties": (
            decreases / (decreases + increases) if decreases + increases else None
        ),
        "exact_two_sided_sign_test_p": exact_sign_test(
            decreases, decreases + increases
        ),
        "paired_scores": paired,
        "provenance": {
            "pre1900_evaluation": str(pre1900_path),
            "pre1900_evaluation_sha256": sha256(pre1900_path),
            "pre1950_evaluation": str(pre1950_path),
            "pre1950_evaluation_sha256": sha256(pre1950_path),
            "bootstrap_repetitions": args.bootstrap,
            "seed": args.seed,
        },
        "interpretation_guardrail": (
            "Positive collapse means the same 1901-1950 family became less anomalous "
            "after entering the admissible knowledge horizon; it is not a claim of "
            "causal discovery or successor-law recovery."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: result[key] for key in (
            "status", "families", "pre1900_median", "pre1950_median",
            "median_collapse", "median_collapse_ci95",
            "fraction_decreased_among_non_ties", "exact_two_sided_sign_test_p",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
