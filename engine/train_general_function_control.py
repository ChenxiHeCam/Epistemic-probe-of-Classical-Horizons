"""Train a capacity/data-matched non-historical normal-function control.

The control receives globally valid smooth analytic curves only.  It contains
no physics provenance and no piecewise breakdown, step, kink, splice,
saturation label or anomaly supervision.  Its sole purpose is to test whether
the strict pre-1900 encoder adds anything beyond generic function pretraining.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "sr_model"))
from models.encoders import DataEncoder


AUDITED_DATA = ROOT / "data" / "pre1900_audited_pointclouds.npz"
CONTROL_DATA = ROOT / "data" / "general_function_control_pointclouds.npz"
CHECKPOINT = ROOT / "data" / "encoder_general_function_control_zero_units.pt"
RESULT = ROOT / "results" / "general_function_control_training.json"
SEED = 20260905
MAXV = 16
DIML = 6
D_MODEL = 128
N_ISAB = 3
STEPS = 1000
BATCH = 32
SUBSAMPLE = 64
POINTS = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_curve(index: int, record_index: int, x: np.ndarray,
                 signature: np.ndarray) -> np.ndarray:
    """Realize a matched signature using one globally valid analytic curve."""
    p = 0.45 + 0.11 * (index % 7) + 0.015 * record_index
    q = p + 0.8 + 0.05 * (index % 5)
    c = 0.18 + 0.06 * (index % 5)
    b = 0.045 + 0.012 * (index % 6)
    y = 0.25 + (0.08 + 0.01 * (index % 4)) * x
    if signature[0]:
        y += 0.18 * np.exp(b * x)
    if signature[1]:
        y += 0.24 * np.log1p((0.35 + b) * x)
    if signature[2]:
        y += 0.12 * np.sin((0.45 + 0.04 * (index % 5)) * x)
    if signature[3]:
        y += 0.30 * np.sqrt(x + c)
    if signature[4]:
        y += 0.16 * (x + c) ** p
    if signature[5]:
        y += 0.22 / (x + c) ** min(q, 2.4)
    if signature[7]:
        y += 0.08 * np.sinh(min(b, 0.08) * x)
    extra_terms = int(round(float(signature[6]) * 6))
    for term in range(extra_terms):
        y += (0.018 / (term + 1)) * x ** (2 + term)
    if not np.any(signature):
        y += (0.012 + 0.001 * (index % 5)) * x**2
    return y


def build_control_data() -> dict:
    audited = np.load(AUDITED_DATA, allow_pickle=True)
    split = np.asarray(audited["splits"]).astype(str)
    families = np.asarray(audited["family_ids"]).astype(str)
    training_families = sorted(set(families[split == "train"]),
                               key=lambda family: (-np.sum((split == "train") & (families == family)), family))
    signature_profiles = [
        np.asarray(audited["sigs"][(split == "train") & (families == family)], np.float32)
        for family in training_families
    ]
    size_profile = [len(profile) for profile in signature_profiles]
    assert len(size_profile) == 40 and sum(size_profile) == 205
    rng = np.random.default_rng(SEED)
    clouds, identifiers, signatures, family_ids = [], [], [], []
    for family_index, profile in enumerate(signature_profiles):
        family = f"generic_global_{family_index:02d}"
        for record_index, signature in enumerate(profile):
            x = np.sort(rng.uniform(0.15, 6.0, POINTS))
            y = family_curve(family_index, record_index, x, signature)
            amplitude = float(np.exp(rng.uniform(-2.0, 2.0)))
            offset = float(rng.uniform(0.0, 0.35) * max(np.std(y), 1e-8))
            y = y - min(float(np.min(y)), 0.0) + 0.05 * max(np.std(y), 1e-8)
            y = amplitude * y + offset
            y += rng.normal(0, 0.005 * max(np.std(y), 1e-8), POINTS)
            assert np.all(np.isfinite(y)) and np.std(y) > 0
            clouds.append(np.stack([x, y], axis=1).astype(np.float32))
            identifiers.append(f"{family}_r{record_index:02d}")
            signatures.append(signature)
            family_ids.append(family)
    np.savez_compressed(
        CONTROL_DATA,
        X=np.asarray(clouds, np.float32),
        ids=np.asarray(identifiers),
        sigs=np.asarray(signatures, np.float32),
        dimsxy=np.zeros((len(clouds), 2, DIML), np.float32),
        family_ids=np.asarray(family_ids),
        splits=np.asarray(["train"] * len(clouds)),
    )
    return {
        "records": len(clouds),
        "families": len(set(family_ids)),
        "family_size_profile": size_profile,
        "signature_rows_exactly_matched": True,
        "unique_signatures": len(set(map(tuple, np.asarray(signatures)))),
        "sha256": sha256(CONTROL_DATA),
    }


def main() -> None:
    data_summary = build_control_data()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.set_per_process_memory_fraction(0.70)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    archive = np.load(CONTROL_DATA, allow_pickle=True)
    clouds = torch.tensor(archive["X"], dtype=torch.float32)
    signatures = torch.tensor(archive["sigs"], dtype=torch.float32)
    families = np.asarray(archive["family_ids"]).astype(str)
    family_names = sorted(set(families))
    family_positions = {family: np.flatnonzero(families == family) for family in family_names}

    encoder = DataEncoder(max_vars=MAXV, d=D_MODEL, n_isab=N_ISAB, dim_len=DIML, n_tokens=16,
                          log_feats=True, class_feats=True, robust_norm=True).to(device)
    auxiliary = nn.Linear(D_MODEL, signatures.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(auxiliary.parameters()), lr=3e-4, weight_decay=1e-5
    )
    rng = np.random.default_rng(SEED)

    def sample_indices() -> np.ndarray:
        chosen = rng.choice(family_names, BATCH, replace=False)
        return np.asarray([rng.choice(family_positions[family]) for family in chosen], dtype=int)

    def make_views(indices: np.ndarray):
        points = torch.zeros(2 * BATCH, SUBSAMPLE, MAXV + 1)
        dims = torch.zeros(2 * BATCH, MAXV + 1, DIML)
        for local, index in enumerate(indices):
            cloud = clouds[index]
            for view in range(2):
                selected = torch.randperm(cloud.shape[0])[:SUBSAMPLE]
                subset = cloud[selected].clone()
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
        return points.to(device), variable_mask.to(device), point_mask.to(device), dims.to(device)

    history = []
    encoder.train()
    auxiliary.train()
    for step in range(1, STEPS + 1):
        indices = sample_indices()
        points, variable_mask, point_mask, dims = make_views(indices)
        embedding = nn.functional.normalize(
            encoder(points, variable_mask, point_mask, dims=dims), dim=-1
        )
        logits = embedding @ embedding.T / 0.1
        logits.fill_diagonal_(-1e9)
        target = (torch.arange(embedding.shape[0], device=device) + BATCH) % embedding.shape[0]
        contrastive = nn.functional.cross_entropy(logits, target)
        target_signature = signatures[indices].repeat(2, 1).to(device)
        signature_loss = nn.functional.binary_cross_entropy_with_logits(
            auxiliary(embedding), target_signature
        )
        loss = contrastive + 0.5 * signature_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "contrastive": float(contrastive.item()),
                   "signature": float(signature_loss.item()), "total": float(loss.item())}
            history.append(row)
            print(json.dumps(row), flush=True)

    payload = {
        "state": encoder.state_dict(),
        "auxiliary_state": auxiliary.state_dict(),
        "d": D_MODEL,
        "n_isab": N_ISAB,
        "max_vars": MAXV,
        "dim_len": DIML,
        "training_data_sha256": sha256(CONTROL_DATA),
        "seed": SEED,
        "training_records": len(clouds),
        "training_families": len(family_names),
        "steps": STEPS,
        "classical_only": False,
        "normal_function_only": True,
        "uses_dimension_metadata": False,
        "control_role": "capacity/data-matched generic-function representation",
    }
    torch.save(payload, CHECKPOINT)
    result = {
        "status": "completed capacity/data-matched generic normal-function control",
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "data": CONTROL_DATA.name,
        "data_sha256": sha256(CONTROL_DATA),
        "device": str(device),
        "training_records": len(clouds),
        "training_families": len(family_names),
        "matched_to_strict": {
            "records": 205,
            "families": 40,
            "family_size_profile": data_summary["family_size_profile"],
            "signature_rows_exactly_matched": data_summary["signature_rows_exactly_matched"],
            "unique_function_signatures": data_summary["unique_signatures"],
            "architecture": {"d_model": D_MODEL, "n_isab": N_ISAB},
            "optimization": {"steps": STEPS, "batch": BATCH, "subsample": SUBSAMPLE, "seed": SEED},
        },
        "history": history,
        "exclusions": [
            "no historical or physical provenance",
            "no piecewise breakdown generator",
            "no breakdown/deformation labels",
            "no saturation, step, kink, splice or crossover supervision",
            "high-level eight-bit function-signature distribution exactly matches the strict training set",
        ],
        "claim_boundary": (
            "This is a generic-function representation control, not a historical corpus and not an "
            "additional EPOCH training source."
        ),
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
