"""Audit learned EPOCH components without tuning on historical headline cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "sr_model"))
from models.encoders import DataEncoder

from calibrated_detector_v2 import BREAKDOWN_FAMILIES, CLASSICAL_FAMILIES, _breakdown


MAXV = 16
DIML = 6
DEVICE = "cpu"


def make_encoder(checkpoint=None):
    model = DataEncoder(max_vars=MAXV, d=256, n_isab=6, dim_len=DIML, n_tokens=16,
                        log_feats=True, class_feats=True, robust_norm=True).to(DEVICE)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


@torch.no_grad()
def encode(model, clouds, dimsxy=None, batch_size=64):
    outputs = []
    for start in range(0, len(clouds), batch_size):
        batch = np.asarray(clouds[start : start + batch_size], np.float32)
        b, n, _ = batch.shape
        points = np.zeros((b, n, MAXV + 1), np.float32)
        points[:, :, 0] = batch[:, :, 0]
        points[:, :, MAXV] = batch[:, :, 1]
        variable_mask = np.zeros((b, MAXV), np.float32)
        variable_mask[:, 0] = 1
        point_mask = np.ones((b, n), np.float32)
        dims = np.zeros((b, MAXV + 1, DIML), np.float32)
        if dimsxy is not None:
            dim_batch = np.asarray(dimsxy[start : start + batch_size], np.float32)
            dims[:, 0] = dim_batch[:, 0]
            dims[:, MAXV] = dim_batch[:, 1]
        z = model(torch.from_numpy(points), torch.from_numpy(variable_mask), torch.from_numpy(point_mask),
                  dims=torch.from_numpy(dims))
        z = torch.nn.functional.normalize(z, dim=-1)
        outputs.append(z.cpu().numpy())
    return np.concatenate(outputs)


def knn_scores(reference, query, k=10):
    scores = []
    for start in range(0, len(query), 128):
        distance = 1 - query[start : start + 128] @ reference.T
        nearest = np.partition(distance, k - 1, axis=1)[:, :k]
        scores.extend(np.mean(nearest, axis=1))
    return np.asarray(scores)


def benchmark_clouds(repeats=8, n=120):
    clouds, labels, families = [], [], []
    seed = 0
    for repeat in range(repeats):
        for family, function in CLASSICAL_FAMILIES.items():
            rng = np.random.default_rng(10_000 + seed)
            x = np.sort(rng.uniform(0.15, 6.0, n))
            y = function(x)
            y += rng.normal(0, 0.02 * max(np.std(y), 1e-9), n)
            clouds.append(np.stack([x, y], axis=1)); labels.append(0); families.append(family); seed += 1
        for family in BREAKDOWN_FAMILIES:
            rng = np.random.default_rng(20_000 + seed)
            x = np.sort(rng.uniform(0.15, 6.0, n))
            y = _breakdown(family, x)
            y += rng.normal(0, 0.02 * max(np.std(y), 1e-9), n)
            clouds.append(np.stack([x, y], axis=1)); labels.append(1); families.append(family); seed += 1
    return np.asarray(clouds, np.float32), np.asarray(labels), families


def deform(cloud, other, kind, rng):
    values = cloud[np.argsort(cloud[:, 0])].copy()
    x, y = values[:, 0], values[:, 1]
    n = len(x)
    split = int(rng.uniform(0.45, 0.75) * n)
    scale = max(np.std(y), 1e-9)
    x0 = x[split]
    if kind == "saturation":
        y[split:] = y[split] + 0.05 * scale * rng.standard_normal(n - split)
    elif kind == "step":
        y[split:] -= np.sign(y[split] + 1e-9) * rng.uniform(1.5, 4.0) * scale
    elif kind == "kink":
        slope = (y[split] - y[max(split - 5, 0)]) / (x[split] - x[max(split - 5, 0)] + 1e-12)
        y[split:] -= rng.uniform(1.0, 3.0) * slope * (x[split:] - x0)
    elif kind == "splice":
        other_y = other[np.argsort(other[:, 0]), 1][: n - split]
        other_y = (other_y - np.mean(other_y)) / (np.std(other_y) + 1e-9) * scale + y[split]
        y[split:] = other_y
    elif kind == "rolloff":
        width = rng.uniform(0.1, 0.5) * (x[-1] - x0 + 1e-9)
        y /= 1 + (np.maximum(x - x0, 0) / width) ** rng.uniform(1.5, 3.0)
    elif kind == "crossover":
        other_y = other[np.argsort(other[:, 0]), 1][:n]
        other_y = (other_y - np.mean(other_y)) / (np.std(other_y) + 1e-9) * scale + np.mean(y)
        width = rng.uniform(0.05, 0.25) * (x[-1] - x[0])
        blend = 1 / (1 + np.exp(-(x - x0) / (width + 1e-9)))
        y = (1 - blend) * y + blend * other_y
    else:
        raise KeyError(kind)
    return np.stack([x, y], axis=1).astype(np.float32)


def main():
    archive = np.load(ROOT / "data" / "classical_pointclouds.npz", allow_pickle=True)
    clouds = archive["X"]
    dimsxy = archive["dimsxy"]
    rng = np.random.default_rng(20260904)
    indices = rng.choice(len(clouds), 2400, replace=False)
    ref_idx, cal_idx = indices[:2000], indices[2000:]
    checkpoint = torch.load(ROOT / "data" / "encoder_1900.pt", map_location=DEVICE)
    trained = make_encoder(checkpoint)
    torch.manual_seed(20260904)
    random_encoder = make_encoder(None)

    z_ref = encode(trained, clouds[ref_idx])
    z_ref_units = encode(trained, clouds[ref_idx], dimsxy[ref_idx])
    z_random_ref = encode(random_encoder, clouds[ref_idx])
    test_clouds, labels, test_families = benchmark_clouds()
    z_test = encode(trained, test_clouds)
    z_random_test = encode(random_encoder, test_clouds)

    scores_trained = knn_scores(z_ref, z_test)
    scores_random = knn_scores(z_random_ref, z_random_test)
    scores_unit_mismatch = knn_scores(z_ref_units, z_test)
    covariance = LedoitWolf().fit(z_ref)
    scores_covariance = covariance.mahalanobis(z_test)
    one_class = {
        "pretrained_knn_zero_metadata": float(roc_auc_score(labels, scores_trained)),
        "random_encoder_knn_zero_metadata": float(roc_auc_score(labels, scores_random)),
        "pretrained_knn_unit_reference_zero_unit_test": float(roc_auc_score(labels, scores_unit_mismatch)),
        "pretrained_shrinkage_mahalanobis_zero_metadata": float(roc_auc_score(labels, scores_covariance)),
    }
    print("one-class", one_class)

    deformation_families = ["saturation", "step", "kink", "splice", "rolloff", "crossover"]
    train_base = clouds[ref_idx[:600]]
    test_base = clouds[cal_idx[:200]]
    z_clean_train = encode(trained, train_base)
    z_clean_test = encode(trained, test_base)
    train_deformed, test_deformed = {}, {}
    for family_index, family in enumerate(deformation_families):
        train_values, test_values = [], []
        local_rng = np.random.default_rng(30_000 + family_index)
        for index, cloud in enumerate(train_base):
            other = clouds[ref_idx[(index + 601 + family_index) % len(ref_idx)]]
            train_values.append(deform(cloud, other, family, local_rng))
        for index, cloud in enumerate(test_base):
            other = clouds[cal_idx[(index + 201 + family_index) % len(cal_idx)]]
            test_values.append(deform(cloud, other, family, local_rng))
        train_deformed[family] = encode(trained, np.asarray(train_values))
        test_deformed[family] = encode(trained, np.asarray(test_values))
        print("encoded deformation", family)

    leave_one_out = {}
    for held_out in deformation_families:
        negative = np.concatenate([train_deformed[family] for family in deformation_families if family != held_out])
        x_train = np.concatenate([z_clean_train, negative])
        y_train = np.concatenate([np.zeros(len(z_clean_train)), np.ones(len(negative))])
        classifier = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(x_train, y_train)
        x_test = np.concatenate([z_clean_test, test_deformed[held_out]])
        y_test = np.concatenate([np.zeros(len(z_clean_test)), np.ones(len(test_deformed[held_out]))])
        score = classifier.predict_proba(x_test)[:, 1]
        leave_one_out[held_out] = float(roc_auc_score(y_test, score))
        print("held out", held_out, leave_one_out[held_out])

    result = {
        "status": "checkpoint audit; training corpus failed period-purity audit",
        "benchmark": {"n": len(labels), "breakdown_families": BREAKDOWN_FAMILIES,
                      "classical_families": list(CLASSICAL_FAMILIES)},
        "one_class_auroc": one_class,
        "leave_one_deformation_family_out_auroc": leave_one_out,
        "leave_one_out_mean": float(np.mean(list(leave_one_out.values()))),
        "metadata_note": (
            "The checkpoint was trained with SI metadata, but the legacy evaluation passes zero metadata. "
            "Synthetic test functions do not have audited SI dimensions, so a valid with-vs-without-units test cannot be claimed."
        ),
    }
    path = ROOT / "results" / "learned_ablation_suite.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
