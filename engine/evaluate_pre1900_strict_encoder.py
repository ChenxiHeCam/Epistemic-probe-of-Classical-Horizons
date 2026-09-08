"""Evaluate source-audited classical-only encoders as one-class detectors."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "sr_model"))
from models.encoders import DataEncoder
from calibrated_detector_v2 import BREAKDOWN_FAMILIES, CLASSICAL_FAMILIES, _breakdown


DATA = ROOT / "data" / "pre1900_audited_pointclouds.npz"
OUTPUT = ROOT / "results" / "pre1900_strict_one_class.json"
MAXV = 16
DIML = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(path: Path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    current_data_sha256 = hashlib.sha256(DATA.read_bytes()).hexdigest()
    training_data_sha256 = checkpoint.get("training_data_sha256")
    if training_data_sha256 != current_data_sha256:
        raise RuntimeError(
            f"checkpoint/data mismatch for {path.name}: trained on {training_data_sha256}, "
            f"current audited corpus is {current_data_sha256}; retrain before evaluation"
        )
    model = DataEncoder(
        max_vars=int(checkpoint["max_vars"]), d=int(checkpoint["d"]),
        n_isab=int(checkpoint["n_isab"]), dim_len=int(checkpoint["dim_len"]),
        n_tokens=16, log_feats=True, class_feats=True, robust_norm=True,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def encode(model, clouds, dimsxy=None, batch_size=64):
    outputs = []
    for start in range(0, len(clouds), batch_size):
        batch = np.asarray(clouds[start:start + batch_size], np.float32)
        b, n, _ = batch.shape
        points = np.zeros((b, n, MAXV + 1), np.float32)
        points[:, :, 0] = batch[:, :, 0]
        points[:, :, MAXV] = batch[:, :, 1]
        variable_mask = np.zeros((b, MAXV), np.float32); variable_mask[:, 0] = 1
        point_mask = np.ones((b, n), np.float32)
        dims = np.zeros((b, MAXV + 1, DIML), np.float32)
        if dimsxy is not None:
            part = np.asarray(dimsxy[start:start + batch_size], np.float32)
            dims[:, 0] = part[:, 0]
            dims[:, MAXV] = part[:, 1]
        z = model(
            torch.from_numpy(points).to(DEVICE),
            torch.from_numpy(variable_mask).to(DEVICE),
            torch.from_numpy(point_mask).to(DEVICE),
            dims=torch.from_numpy(dims).to(DEVICE),
        )
        outputs.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(outputs)


def knn_scores(reference, query, k=10):
    k = min(k, len(reference))
    distance = 1 - query @ reference.T
    return np.mean(np.partition(distance, k - 1, axis=1)[:, :k], axis=1)


def benchmark_clouds(repeats=8, n=120):
    clouds, labels, families = [], [], []
    seed = 0
    for repeat in range(repeats):
        for family, function in CLASSICAL_FAMILIES.items():
            rng = np.random.default_rng(10_000 + seed)
            x = np.sort(rng.uniform(0.15, 6.0, n)); y = function(x)
            y += rng.normal(0, 0.02 * max(np.std(y), 1e-9), n)
            clouds.append(np.stack([x, y], axis=1)); labels.append(0); families.append("C:" + family); seed += 1
        for family in BREAKDOWN_FAMILIES:
            rng = np.random.default_rng(20_000 + seed)
            x = np.sort(rng.uniform(0.15, 6.0, n)); y = _breakdown(family, x)
            y += rng.normal(0, 0.02 * max(np.std(y), 1e-9), n)
            clouds.append(np.stack([x, y], axis=1)); labels.append(1); families.append("B:" + family); seed += 1
    return np.asarray(clouds, np.float32), np.asarray(labels), np.asarray(families)


def clustered_auc_ci(labels, scores, families, seed=20260905, repetitions=2000):
    rng = np.random.default_rng(seed)
    class_families = {
        label: sorted(set(families[labels == label])) for label in (0, 1)
    }
    values = []
    for _ in range(repetitions):
        chosen_indices = []
        for label in (0, 1):
            group = class_families[label]
            chosen = rng.choice(group, len(group), replace=True)
            for family in chosen:
                chosen_indices.extend(np.flatnonzero((labels == label) & (families == family)))
        chosen_indices = np.asarray(chosen_indices, int)
        values.append(roc_auc_score(labels[chosen_indices], scores[chosen_indices]))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def evaluate_variant(name, checkpoint_path, archive, synthetic, labels, synthetic_families):
    if checkpoint_path is None:
        torch.manual_seed(20260905)
        model = DataEncoder(max_vars=MAXV, d=128, n_isab=3, dim_len=DIML, n_tokens=16,
                            log_feats=True, class_feats=True, robust_norm=True).to(DEVICE)
        model.eval()
        checkpoint = {"uses_dimension_metadata": False}
    else:
        model, checkpoint = load_encoder(checkpoint_path)
    split = np.asarray(archive["splits"]).astype(str)
    use_units = bool(checkpoint["uses_dimension_metadata"])
    dims = archive["dimsxy"] if use_units else None
    z_all = encode(model, archive["X"], dims)
    z_reference = z_all[split == "train"]
    z_synthetic = encode(model, synthetic, None)
    knn = knn_scores(z_reference, z_synthetic)
    covariance = LedoitWolf().fit(z_reference)
    mahalanobis = covariance.mahalanobis(z_synthetic)

    result = {"checkpoint": checkpoint_path.name if checkpoint_path else None,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path else None,
              "uses_dimension_metadata": use_units, "synthetic": {}}
    for score_name, scores in (("knn", knn), ("shrinkage_mahalanobis", mahalanobis)):
        auc = float(roc_auc_score(labels, scores))
        result["synthetic"][score_name] = {
            "auroc": auc,
            "family_clustered_95_ci": clustered_auc_ci(labels, scores, synthetic_families),
        }

    validation_scores = knn_scores(z_reference, z_all[split == "validation"])
    validation_family_count = len(set(
        np.asarray(archive["family_ids"])[split == "validation"].astype(str)
    ))
    test_scores = knn_scores(z_reference, z_all[split == "test"])
    breakdown_scores = knn[labels == 1]
    threshold = float(np.quantile(validation_scores, 0.95, method="higher"))
    result["frozen_from_small_validation"] = {
        "threshold": threshold,
        "validation_n": len(validation_scores),
        "test_classical_n": len(test_scores),
        "breakdown_n": len(breakdown_scores),
        "test_classical_fpr": float(np.mean(test_scores > threshold)),
        "synthetic_breakdown_tpr": float(np.mean(breakdown_scores > threshold)),
        "warning": (
            f"Only {len(validation_scores)} validation records from "
            f"{validation_family_count} families; this operating point is exploratory."
        ),
    }

    historical_scores = np.concatenate([test_scores, breakdown_scores])
    historical_labels = np.concatenate([np.zeros(len(test_scores)), np.ones(len(breakdown_scores))])
    historical_families = np.concatenate([
        np.char.add("H:", np.asarray(archive["family_ids"])[split == "test"].astype(str)),
        synthetic_families[labels == 1],
    ])
    result["heldout_historical_classical_vs_synthetic_breakdown"] = {
        "knn_auroc": float(roc_auc_score(historical_labels, historical_scores)),
        "family_clustered_95_ci": clustered_auc_ci(
            historical_labels, historical_scores, historical_families, seed=20260906
        ),
    }
    print(name, json.dumps(result, indent=2), flush=True)
    return result


def main():
    archive = np.load(DATA, allow_pickle=True)
    synthetic, labels, families = benchmark_clouds()
    variants = {
        "with_units": ROOT / "data" / "encoder_pre1900_strict_with_units.pt",
        "zero_units": ROOT / "data" / "encoder_pre1900_strict_zero_units.pt",
        "random_zero_units": None,
    }
    results = {
        name: evaluate_variant(name, path, archive, synthetic, labels, families)
        for name, path in variants.items()
    }
    output = {
        "status": "classical-only source-audited sanity evaluation",
        "audited_reference": {
            "records": len(archive["X"]),
            "families": len(set(np.asarray(archive["family_ids"]).astype(str))),
            "records_by_split": {
                str(name): int(count) for name, count in zip(
                    *np.unique(np.asarray(archive["splits"]).astype(str), return_counts=True)
                )
            },
            "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        },
        "benchmark": {"n": len(labels), "classical_families": list(CLASSICAL_FAMILIES),
                      "breakdown_families": list(BREAKDOWN_FAMILIES)},
        "variants": results,
        "claim_boundary": (
            "No breakdown examples enter training. Results measure small-corpus one-class transfer; "
            "they do not establish a foundation-model-scale learned claim."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
