"""Train a small classical-only encoder on the source-audited pre-1900 subset.

This is a leakage-free sanity experiment, not a foundation-model-scale run.
No breakdown or deformation examples enter optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "sr_model"))
from models.encoders import DataEncoder


DATA = ROOT / "data" / "pre1900_audited_pointclouds.npz"
SEED = 20260905
MAXV = 16
DIML = 6
D_MODEL = 128
N_ISAB = 3
STEPS = 1000
BATCH = 32
SUBSAMPLE = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-units", action="store_true",
                        help="train the matched ablation with all dimension metadata set to zero")
    args = parser.parse_args()
    variant = "zero_units" if args.zero_units else "with_units"
    checkpoint = ROOT / "data" / f"encoder_pre1900_strict_{variant}.pt"
    result_path = ROOT / "results" / f"pre1900_strict_encoder_training_{variant}.json"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.set_per_process_memory_fraction(0.70)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    archive = np.load(DATA, allow_pickle=True)
    split = np.asarray(archive["splits"]).astype(str)
    train_positions = np.flatnonzero(split == "train")
    clouds = torch.tensor(archive["X"][train_positions], dtype=torch.float32)
    signatures = torch.tensor(archive["sigs"][train_positions], dtype=torch.float32)
    dimsxy = torch.tensor(archive["dimsxy"][train_positions], dtype=torch.float32)
    families = np.asarray(archive["family_ids"])[train_positions].astype(str)
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
        chosen_families = rng.choice(family_names, BATCH, replace=False)
        return np.asarray([rng.choice(family_positions[family]) for family in chosen_families], dtype=int)

    def make_views(indices: np.ndarray):
        point_count = clouds.shape[1]
        points = torch.zeros(2 * BATCH, SUBSAMPLE, MAXV + 1)
        dims = torch.zeros(2 * BATCH, MAXV + 1, DIML)
        for local, index in enumerate(indices):
            cloud = clouds[index]
            for view in range(2):
                selected = torch.randperm(point_count)[:SUBSAMPLE]
                subset = cloud[selected].clone()
                subset[:, 0] *= float(np.exp(rng.uniform(-1.5, 1.5)))
                subset[:, 1] *= float(np.exp(rng.uniform(-1.5, 1.5)))
                subset += torch.randn_like(subset) * (0.002 * subset.std(dim=0, keepdim=True).clamp_min(1e-8))
                row = view * BATCH + local
                points[row, :, 0] = subset[:, 0]
                points[row, :, MAXV] = subset[:, 1]
                if not args.zero_units:
                    dims[row, 0] = dimsxy[index, 0]
                    dims[row, MAXV] = dimsxy[index, 1]
        variable_mask = torch.zeros(2 * BATCH, MAXV); variable_mask[:, 0] = 1
        point_mask = torch.ones(2 * BATCH, SUBSAMPLE)
        return points.to(device), variable_mask.to(device), point_mask.to(device), dims.to(device)

    history = []
    encoder.train(); auxiliary.train()
    for step in range(1, STEPS + 1):
        indices = sample_indices()
        points, variable_mask, point_mask, dims = make_views(indices)
        embedding = nn.functional.normalize(
            encoder(points, variable_mask, point_mask, dims=dims), dim=-1
        )
        local_batch = embedding.shape[0]
        batch_logits = embedding @ embedding.T / 0.1
        batch_logits.fill_diagonal_(-1e9)
        logits = batch_logits
        target = (torch.arange(local_batch, device=device) + BATCH) % local_batch
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
            item = {"step": step, "contrastive": float(contrastive.item()),
                    "signature": float(signature_loss.item()), "total": float(loss.item())}
            history.append(item)
            print(json.dumps(item), flush=True)

    payload = {
        "state": encoder.state_dict(), "auxiliary_state": auxiliary.state_dict(),
        "d": D_MODEL, "n_isab": N_ISAB, "max_vars": MAXV, "dim_len": DIML,
        "training_data_sha256": sha256(DATA), "seed": SEED,
        "training_records": len(train_positions), "training_families": len(family_names),
        "steps": STEPS, "classical_only": True, "uses_dimension_metadata": not args.zero_units,
    }
    torch.save(payload, checkpoint)
    result = {
        "status": "completed classical-only source-audited pre-1900 sanity training",
        "variant": variant,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "data": DATA.name,
        "data_sha256": sha256(DATA),
        "device": str(device),
        "training_records": len(train_positions),
        "training_families": len(family_names),
        "optimization": {"steps": STEPS, "batch": BATCH, "subsample": SUBSAMPLE,
                         "d_model": D_MODEL, "n_isab": N_ISAB, "seed": SEED},
        "history": history,
        "claim_boundary": (
            "No breakdown examples were used, but the small audited corpus makes this a sanity experiment, "
            "not evidence for a foundation-scale pre-1900 representation."
        ),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
