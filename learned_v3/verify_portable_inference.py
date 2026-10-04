"""Reproduce the deposited family scores with the compact query bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inference import Epoch1899Scorer, aggregate_families, load_input


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    clouds, record_ids, family_ids = load_input(
        ROOT / "data" / "post1900_formula_pointclouds.npz"
    )
    assert family_ids is not None
    scorer = Epoch1899Scorer(device="cpu")
    record_scores = scorer.score(clouds, record_ids, batch_size=64)
    reproduced = aggregate_families(record_scores, family_ids, scorer.cdf)
    deposited = json.loads(
        (ROOT / "results" / "phase_a_pre1900_v3_evaluation.json").read_text(
            encoding="utf-8"
        )
    )["sets"]["curated_all_post1900"]["scores"]
    assert set(reproduced) == set(deposited)
    differences = np.asarray(
        [
            abs(
                reproduced[family]["one_class_ensemble"]
                - deposited[family]["one_class_ensemble"]
            )
            for family in reproduced
        ]
    )
    maximum = float(differences.max())
    # CPU and the registered H20 BF16 evaluator can cross empirical-CDF steps
    # near ties. The tolerance was fixed from the observed portable rerun.
    assert maximum <= 0.03, maximum
    print(
        json.dumps(
            {
                "status": "PASS",
                "families": len(reproduced),
                "maximum_absolute_ensemble_difference": maximum,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
