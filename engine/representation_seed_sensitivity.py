"""Three-paired-seed sensitivity for strict versus generic pretraining.

The canonical checkpoints remain untouched.  Each seed initializes the same
encoder twice and changes only the normal-training corpus.  Evaluation uses a
fixed view stream and the same 24 synthetic mechanism families.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "sr_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoders import DataEncoder
from evaluate_pre1900_strict_encoder import benchmark_clouds, encode, knn_scores
from evaluate_pre1900_cross_family import classical_views, grouped_medians, record_scores


STRICT_DATA = ROOT / "data" / "pre1900_audited_pointclouds.npz"
GENERIC_DATA = ROOT / "data" / "general_function_control_pointclouds.npz"
OUTPUT = ROOT / "results" / "representation_seed_sensitivity.json"
SEEDS = [20260905, 20260906, 20260907]
EVALUATION_SEED = 20261900
MAXV = 16
DIML = 6
D_MODEL = 128
N_ISAB = 3
STEPS = 1000
BATCH = 32
SUBSAMPLE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_arrays(archive: np.lib.npyio.NpzFile):
    split = np.asarray(archive["splits"]).astype(str)
    keep = split == "train"
    return (
        np.asarray(archive["X"][keep], np.float32),
        np.asarray(archive["sigs"][keep], np.float32),
        np.asarray(archive["family_ids"][keep]).astype(str),
    )


def train_one(cloud_values: np.ndarray, signature_values: np.ndarray,
              family_values: np.ndarray, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    clouds = torch.tensor(cloud_values, dtype=torch.float32)
    signatures = torch.tensor(signature_values, dtype=torch.float32)
    family_names = sorted(set(family_values))
    family_positions = {
        family: np.flatnonzero(family_values == family) for family in family_names
    }
    assert len(clouds) == 205 and len(family_names) == 40
    model = DataEncoder(max_vars=MAXV, d=D_MODEL, n_isab=N_ISAB, dim_len=DIML,
                        n_tokens=16, log_feats=True, class_feats=True,
                        robust_norm=True).to(DEVICE)
    auxiliary = nn.Linear(D_MODEL, signatures.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(auxiliary.parameters()),
        lr=3e-4, weight_decay=1e-5,
    )
    rng = np.random.default_rng(seed)
    final = None
    model.train()
    auxiliary.train()
    for _step in range(1, STEPS + 1):
        selected_families = rng.choice(family_names, BATCH, replace=False)
        indices = np.asarray([
            rng.choice(family_positions[family]) for family in selected_families
        ], dtype=int)
        points = torch.zeros(2 * BATCH, SUBSAMPLE, MAXV + 1)
        dims = torch.zeros(2 * BATCH, MAXV + 1, DIML)
        for local, index in enumerate(indices):
            cloud = clouds[index]
            for view in range(2):
                chosen = torch.randperm(cloud.shape[0])[:SUBSAMPLE]
                subset = cloud[chosen].clone()
                subset[:, 0] *= float(np.exp(rng.uniform(-1.5, 1.5)))
                subset[:, 1] *= float(np.exp(rng.uniform(-1.5, 1.5)))
                subset += torch.randn_like(subset) * (
                    0.002 * subset.std(dim=0, keepdim=True).clamp_min(1e-8)
                )
                row = view * BATCH + local
                points[row, :, 0] = subset[:, 0]
                points[row, :, MAXV] = subset[:, 1]
        variable_mask = torch.zeros(2 * BATCH, MAXV)
        variable_mask[:, 0] = 1
        point_mask = torch.ones(2 * BATCH, SUBSAMPLE)
        embedding = nn.functional.normalize(
            model(points.to(DEVICE), variable_mask.to(DEVICE), point_mask.to(DEVICE),
                  dims=dims.to(DEVICE)),
            dim=-1,
        )
        logits = embedding @ embedding.T / 0.1
        logits.fill_diagonal_(-1e9)
        target = (torch.arange(2 * BATCH, device=DEVICE) + BATCH) % (2 * BATCH)
        contrastive = nn.functional.cross_entropy(logits, target)
        target_signature = signatures[indices].repeat(2, 1).to(DEVICE)
        signature_loss = nn.functional.binary_cross_entropy_with_logits(
            auxiliary(embedding), target_signature
        )
        loss = contrastive + 0.5 * signature_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final = (float(contrastive.item()), float(signature_loss.item()), float(loss.item()))
    model.eval()
    return model, {
        "contrastive": final[0], "signature": final[1], "total": final[2]
    }


def evaluate(model, train_clouds, train_families, heldout_archive):
    rng = np.random.default_rng(EVALUATION_SEED)
    views, owners = [], []
    for index, cloud in enumerate(train_clouds):
        generated = classical_views(cloud, rng)
        views.extend(generated)
        owners.extend([index] * len(generated))
    embeddings = encode(model, views)
    owners = np.asarray(owners)
    calibration = []
    for family in sorted(set(train_families)):
        reference = embeddings[train_families[owners] != family]
        positions = np.flatnonzero(train_families == family)
        per_record = [
            float(np.median(knn_scores(reference, embeddings[owners == index])))
            for index in positions
        ]
        calibration.append(float(np.median(per_record)))
    threshold = float(np.quantile(np.asarray(calibration), 0.95, method="higher"))

    synthetic, labels, synthetic_families = benchmark_clouds(n=64)
    synthetic_scores = knn_scores(embeddings, encode(model, synthetic))
    grouped = grouped_medians(synthetic_scores, synthetic_families)
    names = sorted(grouped)
    family_scores = np.asarray([grouped[name] for name in names])
    family_labels = np.asarray([name.startswith("B:") for name in names], int)

    heldout_split = np.asarray(heldout_archive["splits"]).astype(str)
    heldout_families = np.asarray(heldout_archive["family_ids"]).astype(str)
    test_positions = np.flatnonzero(heldout_split == "test")
    test_scores = record_scores(
        model, embeddings, heldout_archive["X"][test_positions], rng
    )
    test_grouped = grouped_medians(test_scores, heldout_families[test_positions])
    return {
        "dataset_auroc": float(roc_auc_score(labels, synthetic_scores)),
        "family_auroc": float(roc_auc_score(family_labels, family_scores)),
        "breakdown_family_tpr": float(np.mean(family_scores[family_labels == 1] > threshold)),
        "synthetic_classical_family_fpr": float(np.mean(family_scores[family_labels == 0] > threshold)),
        "cited_test_family_fpr": float(np.mean(np.asarray(list(test_grouped.values())) > threshold)),
        "threshold": threshold,
    }


def summarize(rows, key):
    values = np.asarray([row[key] for row in rows], float)
    return {
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70)
    strict_archive = np.load(STRICT_DATA, allow_pickle=True)
    generic_archive = np.load(GENERIC_DATA, allow_pickle=True)
    strict_arrays = training_arrays(strict_archive)
    generic_arrays = training_arrays(generic_archive)
    assert Counter(map(tuple, strict_arrays[1])) == Counter(map(tuple, generic_arrays[1]))
    assert sorted(Counter(strict_arrays[2]).values()) == sorted(Counter(generic_arrays[2]).values())

    rows = []
    for seed in SEEDS:
        strict_model, strict_loss = train_one(*strict_arrays, seed)
        strict_metrics = evaluate(strict_model, strict_arrays[0], strict_arrays[2], strict_archive)
        del strict_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        generic_model, generic_loss = train_one(*generic_arrays, seed)
        generic_metrics = evaluate(generic_model, generic_arrays[0], generic_arrays[2], strict_archive)
        del generic_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row = {
            "seed": seed,
            "strict": strict_metrics,
            "generic": generic_metrics,
            "strict_minus_generic_family_auroc": (
                strict_metrics["family_auroc"] - generic_metrics["family_auroc"]
            ),
            "final_training_loss": {"strict": strict_loss, "generic": generic_loss},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "status": "completed paired training-seed sensitivity; canonical checkpoints unchanged",
        "seeds": SEEDS,
        "evaluation_seed": EVALUATION_SEED,
        "matched": {
            "records": 205,
            "families": 40,
            "family_size_profile": True,
            "function_signature_histogram": True,
            "architecture": True,
            "optimization": True,
            "paired_initialization_seed": True,
        },
        "runs": rows,
        "summary": {
            "strict_family_auroc": summarize([row["strict"] for row in rows], "family_auroc"),
            "generic_family_auroc": summarize([row["generic"] for row in rows], "family_auroc"),
            "strict_minus_generic_family_auroc": summarize(
                rows, "strict_minus_generic_family_auroc"
            ),
            "strict_synthetic_classical_family_fpr": summarize(
                [row["strict"] for row in rows], "synthetic_classical_family_fpr"
            ),
            "generic_synthetic_classical_family_fpr": summarize(
                [row["generic"] for row in rows], "synthetic_classical_family_fpr"
            ),
        },
        "claim_boundary": (
            "Three paired seeds estimate training instability descriptively; n=3 is not a precise "
            "training-population confidence interval, and all evaluations remain synthetic."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
