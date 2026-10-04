"""Metadata-consistent units ablation on held-out cited pre-1900 laws.

Synthetic deformations alter y(x) while retaining each archived input/output SI
dimension vector.  This tests the unit channel without the zero-unit mismatch of
the generic synthetic suite.  The SI vectors are structurally checked archive
metadata, not a new manual dimensional-analysis audit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pre1900_strict_encoder import DATA, encode, knn_scores, load_encoder
from evaluate_pre1900_cross_family import classical_views, grouped_medians
from learned_ablation_suite import deform


OUTPUT = ROOT / "results" / "unit_consistent_deformation_ablation.json"
CHECKPOINTS = {
    "with_units": ROOT / "data" / "encoder_pre1900_strict_with_units.pt",
    "zero_units": ROOT / "data" / "encoder_pre1900_strict_zero_units.pt",
}
DEFORMATIONS = ["saturation", "step", "kink", "splice", "rolloff", "crossover"]
SEED = 20260908


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def view_embeddings(model, clouds, dimsxy, rng, use_units):
    views, view_dims, owners = [], [], []
    for local, cloud in enumerate(clouds):
        generated = classical_views(cloud, rng)
        views.extend(generated)
        owners.extend([local] * len(generated))
        if use_units:
            view_dims.extend([dimsxy[local]] * len(generated))
    return encode(model, views, np.asarray(view_dims) if use_units else None), np.asarray(owners)


def query_scores(model, reference, clouds, dimsxy, rng, use_units):
    embeddings, owners = view_embeddings(model, clouds, dimsxy, rng, use_units)
    return np.asarray([
        float(np.median(knn_scores(reference, embeddings[owners == index])))
        for index in range(len(clouds))
    ])


def source_balanced_vectors(clean, deformed, source_families):
    labels, scores, groups = [], [], []
    for family in sorted(set(source_families)):
        keep = source_families == family
        labels.append(0)
        scores.append(float(np.median(clean[keep])))
        groups.append(family)
        for deformation in DEFORMATIONS:
            labels.append(1)
            scores.append(float(np.median(deformed[deformation][keep])))
            groups.append(family)
    return np.asarray(labels), np.asarray(scores), np.asarray(groups)


def paired_source_bootstrap_delta(vectors, seed=SEED, repetitions=5000):
    labels = vectors["with_units"][0]
    groups = vectors["with_units"][2]
    assert np.array_equal(labels, vectors["zero_units"][0])
    assert np.array_equal(groups, vectors["zero_units"][2])
    families = sorted(set(groups))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        chosen = rng.choice(families, len(families), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == family) for family in chosen])
        deltas.append(
            roc_auc_score(labels[indices], vectors["with_units"][1][indices])
            - roc_auc_score(labels[indices], vectors["zero_units"][1][indices])
        )
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def main() -> None:
    archive = np.load(DATA, allow_pickle=True)
    split = np.asarray(archive["splits"]).astype(str)
    family_ids = np.asarray(archive["family_ids"]).astype(str)
    train_positions = np.flatnonzero(split == "train")
    test_positions = np.flatnonzero(split == "test")
    train_clouds = archive["X"][train_positions]
    train_dims = archive["dimsxy"][train_positions]
    train_families = family_ids[train_positions]
    test_clouds = archive["X"][test_positions]
    test_dims = archive["dimsxy"][test_positions]
    test_families = family_ids[test_positions]
    assert len(test_clouds) == 46 and len(set(test_families)) == 7

    deformation_rng = np.random.default_rng(SEED)
    deformed = {}
    for family_index, kind in enumerate(DEFORMATIONS):
        values = []
        for index, cloud in enumerate(test_clouds):
            other = test_clouds[(index + family_index + 1) % len(test_clouds)]
            values.append(deform(cloud, other, kind, deformation_rng))
        deformed[kind] = np.asarray(values, np.float32)

    variants = {}
    balanced = {}
    for name, checkpoint_path in CHECKPOINTS.items():
        model, checkpoint = load_encoder(checkpoint_path)
        use_units = bool(checkpoint["uses_dimension_metadata"])
        # Identical view/noise stream for the paired metadata comparison.
        rng = np.random.default_rng(SEED)
        train_embeddings, owners = view_embeddings(
            model, train_clouds, train_dims, rng, use_units
        )
        family_calibration = {}
        for family in sorted(set(train_families)):
            reference_loo = train_embeddings[train_families[owners] != family]
            positions = np.flatnonzero(train_families == family)
            per_record = [
                float(np.median(knn_scores(reference_loo, train_embeddings[owners == index])))
                for index in positions
            ]
            family_calibration[family] = float(np.median(per_record))
        threshold = float(np.quantile(
            np.asarray(list(family_calibration.values())), 0.95, method="higher"
        ))
        clean_scores = query_scores(
            model, train_embeddings, test_clouds, test_dims, rng, use_units
        )
        deformation_scores = {
            kind: query_scores(model, train_embeddings, clouds, test_dims, rng, use_units)
            for kind, clouds in deformed.items()
        }
        labels = np.concatenate([
            np.zeros(len(clean_scores)),
            np.ones(len(clean_scores) * len(DEFORMATIONS)),
        ])
        scores = np.concatenate([clean_scores] + [deformation_scores[kind] for kind in DEFORMATIONS])
        balanced[name] = source_balanced_vectors(clean_scores, deformation_scores, test_families)
        balanced_labels, balanced_scores, _ = balanced[name]
        clean_family_scores = grouped_medians(clean_scores, test_families)
        deformed_source_family_scores = {
            kind: grouped_medians(values, test_families)
            for kind, values in deformation_scores.items()
        }
        deformation_family_scores = {
            kind: float(np.median(values)) for kind, values in deformation_scores.items()
        }
        variants[name] = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "uses_dimension_metadata": use_units,
            "threshold": threshold,
            "calibration_source_family_scores": family_calibration,
            "record_auroc": float(roc_auc_score(labels, scores)),
            "source_family_balanced_auroc": float(roc_auc_score(balanced_labels, balanced_scores)),
            "clean_record_fpr": float(np.mean(clean_scores > threshold)),
            "clean_source_family_fpr": float(np.mean(
                np.asarray(list(clean_family_scores.values())) > threshold
            )),
            "deformed_record_tpr": float(np.mean(
                np.concatenate(list(deformation_scores.values())) > threshold
            )),
            "deformed_source_family_cell_tpr": float(np.mean(
                balanced_scores[balanced_labels == 1] > threshold
            )),
            "global_deformation_median_tpr": float(np.mean(
                np.asarray(list(deformation_family_scores.values())) > threshold
            )),
            "pairwise_score_increase": float(np.mean(np.concatenate([
                deformation_scores[kind] > clean_scores for kind in DEFORMATIONS
            ]))),
            "clean_source_family_scores": clean_family_scores,
            "deformed_source_family_scores": deformed_source_family_scores,
            "by_deformation": {
                kind: {
                    "record_auroc_vs_clean": float(roc_auc_score(
                        np.concatenate([np.zeros(len(clean_scores)), np.ones(len(clean_scores))]),
                        np.concatenate([clean_scores, deformation_scores[kind]]),
                    )),
                    "record_tpr": float(np.mean(deformation_scores[kind] > threshold)),
                    "median_score": deformation_family_scores[kind],
                }
                for kind in DEFORMATIONS
            },
        }

    delta = (
        variants["with_units"]["source_family_balanced_auroc"]
        - variants["zero_units"]["source_family_balanced_auroc"]
    )
    result = {
        "status": "completed unit-metadata-consistent held-out-law deformation audit",
        "data_sha256": sha256(DATA),
        "heldout": {
            "records": len(test_clouds),
            "source_law_families": len(set(test_families)),
            "deformation_families": DEFORMATIONS,
            "training_records": len(train_clouds),
            "training_source_law_families": len(set(train_families)),
            "training_records_with_any_nonzero_dimension_metadata": int(np.sum(
                np.any(train_dims != 0, axis=(1, 2))
            )),
            "records_with_any_nonzero_dimension_metadata": int(np.sum(
                np.any(test_dims != 0, axis=(1, 2))
            )),
            "dimension_preservation": "each deformed point cloud retains its source record's x/y SI vector",
        },
        "protocol": {
            "ranking_unit": "one clean median and six deformation medians within each held-out source-law family",
            "calibration_unit": "leave-one-source-law-family-out median over 40 training families",
            "threshold": "95th-percentile higher-order statistic, frozen separately for each checkpoint",
            "paired_bootstrap_unit": "held-out source-law family",
            "bootstrap_repetitions": 5000,
        },
        "variants": variants,
        "with_minus_zero_source_family_balanced_auroc": delta,
        "paired_source_family_bootstrap_95_ci": paired_source_bootstrap_delta(balanced),
        "claim_boundary": (
            "The comparison removes train/test metadata-format mismatch but uses synthetic deformations "
            "and only seven cited test families under one paired training seed. Archived SI vectors were "
            "not manually re-derived for every formula, so this is not a full physical-dimension audit."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
