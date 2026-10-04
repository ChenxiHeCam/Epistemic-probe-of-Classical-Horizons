"""Family-level cross-conformal audit of the strict classical-only encoder.

Calibration units are source-law families. Augmented views are never counted as
independent calibration observations.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pre1900_strict_encoder import DATA, ROOT, benchmark_clouds, encode, knn_scores, load_encoder


CHECKPOINT = ROOT / "data" / "encoder_pre1900_strict_zero_units.pt"
OUTPUT = ROOT / "results" / "pre1900_cross_family_conformal.json"
SEED = 20260905
VIEWS = 8


def classical_views(cloud, rng):
    output = []
    for _ in range(VIEWS):
        indices = np.sort(rng.choice(len(cloud), 64, replace=False))
        values = np.asarray(cloud, float)[indices].copy()
        values[:, 0] *= np.exp(rng.uniform(-1.5, 1.5))
        values[:, 1] *= np.exp(rng.uniform(-1.5, 1.5))
        scale_y = max(np.std(values[:, 1]), 1e-9)
        scale_x = max(np.ptp(values[:, 0]), 1e-9)
        heteroscedastic = 1 + rng.uniform(0, 1) * np.linspace(0, 1, 64)
        values[:, 1] += rng.normal(0, rng.uniform(0, 0.05) * scale_y, 64) * heteroscedastic
        values[:, 0] += rng.normal(0, rng.uniform(0, 0.005) * scale_x, 64)
        output.append(values.astype(np.float32))
    return output


def real_views(cloud, rng):
    count = min(64, len(cloud))
    output = []
    for _ in range(VIEWS):
        indices = np.sort(rng.choice(len(cloud), count, replace=False))
        values = np.asarray(cloud, float)[indices].copy()
        values[:, 0] *= np.exp(rng.uniform(-1.5, 1.5))
        values[:, 1] *= np.exp(rng.uniform(-1.5, 1.5))
        output.append(values.astype(np.float32))
    return output


def record_scores(model, reference, clouds, rng, real=False):
    scores = []
    for cloud in clouds:
        views = real_views(cloud, rng) if real else classical_views(cloud, rng)
        scores.append(float(np.median(knn_scores(reference, encode(model, views)))))
    return np.asarray(scores)


def grouped_medians(scores, families):
    return {
        family: float(np.median(scores[families == family]))
        for family in sorted(set(families))
    }


def bootstrap_family_auc(labels, scores, seed=SEED, repetitions=5000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = np.concatenate([
            rng.choice(np.flatnonzero(labels == label), np.sum(labels == label), replace=True)
            for label in (0, 1)
        ])
        values.append(roc_auc_score(labels[indices], scores[indices]))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def load_real_cases():
    from score_real_cases_pre1900_strict import load_two_columns, michelson

    firas = load_two_columns(ROOT / "data" / "firas_monopole.txt")
    coefficients = [-1.91844, -0.15973, 8.61013, -18.996, 21.9661, -12.7328, 3.54322, -0.3797, 0]
    temperature = np.logspace(np.log10(4), np.log10(300), 120)
    heat_capacity = 10 ** np.clip(
        sum(value * np.log10(temperature) ** power for power, value in enumerate(coefficients)), -30, 30
    )
    boyle = np.asarray(json.loads((ROOT / "data" / "boyle_1662.json").read_text()), np.float32)
    return {
        "FIRAS_blackbody": firas,
        "copper_specific_heat": np.stack([temperature, heat_capacity], axis=1).astype(np.float32),
        "Boyle_1662_control": boyle,
        "Michelson_1879_control": michelson(),
    }


def main():
    archive = np.load(DATA, allow_pickle=True)
    split = np.asarray(archive["splits"]).astype(str)
    families = np.asarray(archive["family_ids"]).astype(str)
    model, checkpoint = load_encoder(CHECKPOINT)
    assert not checkpoint["uses_dimension_metadata"]
    rng = np.random.default_rng(SEED)

    train_indices = np.flatnonzero(split == "train")
    train_views, owners = [], []
    for index in train_indices:
        values = classical_views(archive["X"][index], rng)
        train_views.extend(values); owners.extend([index] * len(values))
    train_embeddings = encode(model, train_views)
    owners = np.asarray(owners)

    calibration = {}
    for family in sorted(set(families[train_indices])):
        reference = train_embeddings[families[owners] != family]
        positions = train_indices[families[train_indices] == family]
        per_record = [
            float(np.median(knn_scores(reference, train_embeddings[owners == index])))
            for index in positions
        ]
        calibration[family] = {"records": len(positions), "median_score": float(np.median(per_record))}
    calibration_values = np.asarray([value["median_score"] for value in calibration.values()])
    threshold = float(np.quantile(calibration_values, 0.95, method="higher"))
    reference = train_embeddings

    heldout = {}
    for split_name in ("validation", "test"):
        positions = np.flatnonzero(split == split_name)
        scores = record_scores(model, reference, archive["X"][positions], rng)
        grouped = grouped_medians(scores, families[positions])
        heldout[split_name] = {
            "records": len(positions),
            "families": len(grouped),
            "record_fpr": float(np.mean(scores > threshold)),
            "family_fpr": float(np.mean(np.asarray(list(grouped.values())) > threshold)),
            "family_scores": grouped,
        }

    synthetic, labels, synthetic_families = benchmark_clouds(n=64)
    synthetic_scores = knn_scores(reference, encode(model, synthetic))
    synthetic_grouped = grouped_medians(synthetic_scores, synthetic_families)
    family_names = sorted(synthetic_grouped)
    family_scores = np.asarray([synthetic_grouped[name] for name in family_names])
    family_labels = np.asarray([name.startswith("B:") for name in family_names], int)
    synthetic_result = {
        "dataset_auroc": float(roc_auc_score(labels, synthetic_scores)),
        "family_auroc": float(roc_auc_score(family_labels, family_scores)),
        "family_auroc_bootstrap_95_ci": bootstrap_family_auc(family_labels, family_scores),
        "breakdown_dataset_tpr_at_threshold": float(np.mean(synthetic_scores[labels == 1] > threshold)),
        "breakdown_family_tpr_at_threshold": float(np.mean(family_scores[family_labels == 1] > threshold)),
        "classical_family_fpr_at_threshold": float(np.mean(family_scores[family_labels == 0] > threshold)),
        "family_scores": synthetic_grouped,
    }

    real_results = {}
    for name, cloud in load_real_cases().items():
        relative_variation = float(np.std(cloud[:, 1]) / (abs(np.mean(cloud[:, 1])) + 1e-12))
        if relative_variation < 0.001:
            real_results[name] = {
                "n": len(cloud), "route": "abstain_from_generic_learned_component",
                "reason": "near-constant response; route to a theory-conditioned constant/noise model",
                "relative_response_variation": relative_variation,
            }
            continue
        score = float(record_scores(model, reference, [cloud], rng, real=True)[0])
        real_results[name] = {
            "n": len(cloud), "route": "generic_learned_component", "score": score,
            "family_conformal_p": float((1 + np.sum(calibration_values >= score)) / (len(calibration_values) + 1)),
            "alert": bool(score > threshold),
            "relative_response_variation": relative_variation,
        }

    output = {
        "status": "completed source-audited classical-only family-level cross-conformal audit",
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "protocol": {
            "training_families": len(calibration), "views_per_record": VIEWS,
            "calibration_unit": "source-law family median",
            "reference_exclusion": "entire calibration family excluded",
            "tail": "strictly greater than 95th-percentile higher-order threshold",
            "threshold": threshold,
        },
        "calibration": calibration,
        "heldout_classical": heldout,
        "synthetic": synthetic_result,
        "real_cases": real_results,
        "claim_boundary": (
            "No breakdown/deformation examples enter training or calibration. The family-level result is a "
            "small-corpus retrospective audit; the real-case learned scores remain exploratory."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
