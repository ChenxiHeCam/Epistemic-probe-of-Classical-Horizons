"""Evaluate the matched generic-function encoder against the strict encoder."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "sr_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoders import DataEncoder
from evaluate_pre1900_strict_encoder import benchmark_clouds, clustered_auc_ci, encode, knn_scores
from evaluate_pre1900_cross_family import classical_views, grouped_medians, bootstrap_family_auc


CONTROL_DATA = ROOT / "data" / "general_function_control_pointclouds.npz"
CHECKPOINT = ROOT / "data" / "encoder_general_function_control_zero_units.pt"
AUDITED_DATA = ROOT / "data" / "pre1900_audited_pointclouds.npz"
STRICT_RESULT = ROOT / "results" / "pre1900_cross_family_conformal.json"
OUTPUT = ROOT / "results" / "general_function_control_evaluation.json"
SEED = 20260905
VIEWS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    assert checkpoint["training_data_sha256"] == sha256(CONTROL_DATA)
    assert checkpoint["normal_function_only"] and not checkpoint["uses_dimension_metadata"]
    model = DataEncoder(
        max_vars=int(checkpoint["max_vars"]), d=int(checkpoint["d"]),
        n_isab=int(checkpoint["n_isab"]), dim_len=int(checkpoint["dim_len"]),
        n_tokens=16, log_feats=True, class_feats=True, robust_norm=True,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


def paired_auc_delta_ci(labels, strict_scores, control_scores, repetitions=5000):
    rng = np.random.default_rng(SEED + 1)
    values = []
    for _ in range(repetitions):
        indices = np.concatenate([
            rng.choice(np.flatnonzero(labels == label), np.sum(labels == label), replace=True)
            for label in (0, 1)
        ])
        values.append(
            roc_auc_score(labels[indices], strict_scores[indices])
            - roc_auc_score(labels[indices], control_scores[indices])
        )
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    control = np.load(CONTROL_DATA, allow_pickle=True)
    model, checkpoint = load_model()
    rng = np.random.default_rng(SEED)
    families = np.asarray(control["family_ids"]).astype(str)

    train_views, owners = [], []
    for index, cloud in enumerate(control["X"]):
        views = classical_views(cloud, rng)
        train_views.extend(views)
        owners.extend([index] * len(views))
    train_embeddings = encode(model, train_views)
    owners = np.asarray(owners)

    calibration = {}
    for family in sorted(set(families)):
        reference = train_embeddings[families[owners] != family]
        positions = np.flatnonzero(families == family)
        record_scores = [
            float(np.median(knn_scores(reference, train_embeddings[owners == index])))
            for index in positions
        ]
        calibration[family] = {"records": len(positions), "median_score": float(np.median(record_scores))}
    calibration_values = np.asarray([row["median_score"] for row in calibration.values()])
    threshold = float(np.quantile(calibration_values, 0.95, method="higher"))
    reference = train_embeddings

    synthetic, labels, synthetic_families = benchmark_clouds(n=64)
    embeddings = encode(model, synthetic)
    knn = knn_scores(reference, embeddings)
    mahalanobis = LedoitWolf().fit(reference).mahalanobis(embeddings)
    grouped = grouped_medians(knn, synthetic_families)
    family_names = sorted(grouped)
    family_scores = np.asarray([grouped[name] for name in family_names])
    family_labels = np.asarray([name.startswith("B:") for name in family_names], int)

    audited = np.load(AUDITED_DATA, allow_pickle=True)
    audited_split = np.asarray(audited["splits"]).astype(str)
    audited_families = np.asarray(audited["family_ids"]).astype(str)
    strict_train = audited_split == "train"
    assert Counter(map(tuple, control["sigs"])) == Counter(map(tuple, audited["sigs"][strict_train]))
    assert sorted(Counter(families).values()) == sorted(Counter(audited_families[strict_train]).values())
    heldout = {}
    for split_name in ("validation", "test"):
        positions = np.flatnonzero(audited_split == split_name)
        scores = knn_scores(reference, encode(model, audited["X"][positions]))
        family_medians = grouped_medians(scores, audited_families[positions])
        heldout[split_name] = {
            "records": len(positions),
            "families": len(family_medians),
            "record_fpr": float(np.mean(scores > threshold)),
            "family_fpr": float(np.mean(np.asarray(list(family_medians.values())) > threshold)),
            "family_scores": family_medians,
        }

    strict = json.loads(STRICT_RESULT.read_text(encoding="utf-8"))
    strict_grouped = strict["synthetic"]["family_scores"]
    strict_scores = np.asarray([strict_grouped[name] for name in family_names])
    strict_auc = float(roc_auc_score(family_labels, strict_scores))
    control_auc = float(roc_auc_score(family_labels, family_scores))
    output = {
        "status": "completed capacity/data-matched generic-function representation control",
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "data": CONTROL_DATA.name,
        "data_sha256": sha256(CONTROL_DATA),
        "protocol": {
            "training_records": int(checkpoint["training_records"]),
            "training_families": int(checkpoint["training_families"]),
            "views_per_record": VIEWS,
            "calibration_unit": "synthetic generator-family median",
            "threshold": threshold,
            "no_anomaly_supervision": True,
            "record_count_match": True,
            "family_size_profile_match": True,
            "function_signature_histogram_match": True,
        },
        "synthetic": {
            "dataset_knn_auroc": float(roc_auc_score(labels, knn)),
            "dataset_knn_family_clustered_95_ci": clustered_auc_ci(labels, knn, synthetic_families),
            "dataset_mahalanobis_auroc": float(roc_auc_score(labels, mahalanobis)),
            "dataset_mahalanobis_family_clustered_95_ci": clustered_auc_ci(
                labels, mahalanobis, synthetic_families, seed=SEED + 2
            ),
            "family_knn_auroc": control_auc,
            "family_knn_bootstrap_95_ci": bootstrap_family_auc(family_labels, family_scores),
            "breakdown_dataset_tpr_at_threshold": float(np.mean(knn[labels == 1] > threshold)),
            "breakdown_family_tpr_at_threshold": float(np.mean(family_scores[family_labels == 1] > threshold)),
            "classical_family_fpr_at_threshold": float(np.mean(family_scores[family_labels == 0] > threshold)),
            "family_scores": grouped,
        },
        "heldout_cited_pre1900": heldout,
        "incremental_comparison": {
            "strict_pre1900_family_auroc": strict_auc,
            "generic_function_family_auroc": control_auc,
            "strict_minus_generic_family_auroc": strict_auc - control_auc,
            "paired_family_bootstrap_95_ci": paired_auc_delta_ci(
                family_labels, strict_scores, family_scores
            ),
            "interpretation": (
                "The paired interval tests ranking increment over generic function pretraining. "
                "It does not compare historical semantics unavailable to a notation-free encoder."
            ),
        },
        "claim_boundary": (
            "This matched control contains no historical provenance and no anomaly supervision. "
            "All comparisons remain synthetic/developmental."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
