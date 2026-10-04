"""Score Paper 2's post-score case expansion with the frozen Phase-A model.

The script reconstructs the published pre-1900 memories and calibration CDF,
checks them against the already reported 1901--1950 curated scores, and then
performs forward inference on the separate case-study expansion.  It never
constructs an optimizer or updates model state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PAPER2 = WORKSPACE / "PAPER2_BOUNDARY"

from epoch_formula_graph_dataset import EpochFormulaGraphStore  # noqa: E402
from evaluate_epoch_horizon_v3 import (  # noqa: E402
    add_frozen_ensemble,
    aggregate_by_key,
    fit_empirical_cdf,
    graph_memories,
    load_model,
    point_formula_memory,
    score_curated_archive,
    score_pointcloud_role,
    sha256,
)
from train_epoch_horizon_v3 import contrastive_identity_lookup  # noqa: E402


CONFIG = ROOT / "EPOCH_RUN_CONFIGS_V2" / "train_pre1900.json"
CHECKPOINT = ROOT / "EPOCH_MODELS_V3" / "phase_a_pre1900_checkpoint_final.pt"
POINT_BUNDLE = ROOT / "EPOCH_RUNTIME_V3" / "pointcloud_full_final"
VIEW_INDEX = ROOT / "EPOCH_POINTCLOUD_VIEWS_V2"
GRAPH_BUNDLE = ROOT / "EPOCH_FORMULA_GRAPHS_V2"
EXPANSION = PAPER2 / "data" / "case_study_expansion_pointclouds.npz"
EXPANSION_REGISTRY = PAPER2 / "data" / "case_study_expansion_registry.json"
VALIDATION_ARCHIVE = ROOT / "EPOCH_TEMPORAL_BENCHMARKS_V2" / "post1900_to_1950.npz"
PUBLISHED = PAPER2 / "results" / "phase_a_pre1900_v3_evaluation.json"
OUTPUT = PAPER2 / "results" / "case_study_expansion_v3_evaluation.json"
CACHE = ROOT / "EPOCH_RUNTIME_V3" / "phase_a_inference_memory.pt"


def to_cpu_memories(memories: dict) -> dict:
    output = {}
    for key, value in memories.items():
        if key == "point_gaussian":
            output[key] = {
                "mean": value["mean"].detach().cpu(),
                "precision": value["precision"].detach().cpu(),
                "shrinkage": float(value["shrinkage"]),
            }
        elif torch.is_tensor(value):
            output[key] = value.detach().cpu()
        else:
            output[key] = value
    return output


def to_device_memories(memories: dict, device: torch.device) -> dict:
    output = {}
    for key, value in memories.items():
        if key == "point_gaussian":
            output[key] = {
                "mean": value["mean"].to(device),
                "precision": value["precision"].to(device),
                "shrinkage": float(value["shrinkage"]),
            }
        elif torch.is_tensor(value):
            output[key] = value.to(device) if value.dtype.is_floating_point else value
        else:
            output[key] = value
    return output


def build_or_load_memory(
    model, graph_store, formula_to_identity, point_config, device,
    batch_size: int, graph_batch_size: int, workers: int, points: int, top_k: int,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    identity = {
        "checkpoint_sha256": sha256(CHECKPOINT),
        "point_manifest_sha256": sha256(POINT_BUNDLE / "manifest.json"),
        "view_manifest_sha256": sha256(VIEW_INDEX / "manifest.json"),
        "graph_manifest_sha256": sha256(GRAPH_BUNDLE / "manifest.json"),
        "points": points, "top_k": top_k,
    }
    if CACHE.exists():
        payload = torch.load(CACHE, map_location="cpu", weights_only=False)
        if payload.get("identity") == identity:
            return to_device_memories(payload["memories"], device), payload["cdf"], {
                "cache_used": True, "cache_sha256": sha256(CACHE), **identity,
            }

    memories = graph_memories(
        model, graph_store, formula_to_identity, point_config["view"],
        "strict", ["pre1900"], device, graph_batch_size,
    )
    point_memory, point_indices = point_formula_memory(
        model, point_config, "strict", ["pre1900"], batch_size,
        points, workers, len(graph_store), device,
    )
    memories["point_formula"] = point_memory
    memories["point_formula_indices"] = point_indices
    from evaluate_epoch_horizon_v3 import shrinkage_gaussian_memory
    memories["point_gaussian"] = shrinkage_gaussian_memory(point_memory)

    calibration_ids, calibration_cloud_scores = score_pointcloud_role(
        model, memories, point_config, "calibration", "strict", ["pre1900"],
        batch_size, points, workers, top_k, device,
    )
    calibration_formula = aggregate_by_key(calibration_ids, calibration_cloud_scores)
    cdf = fit_empirical_cdf(calibration_formula)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "memories": to_cpu_memories(memories), "cdf": cdf}, CACHE)
    return memories, cdf, {"cache_used": False, "cache_sha256": sha256(CACHE), **identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--graph-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reconstruct the published Phase-A inference memory")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    point_config = dict(config["pointcloud"])
    point_config["bundle"] = str(POINT_BUNDLE)
    point_config["view_index"] = str(VIEW_INDEX)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    graph_store = EpochFormulaGraphStore(GRAPH_BUNDLE)
    formula_to_identity, identity_names, verified_table = contrastive_identity_lookup(
        graph_store, POINT_BUNDLE, ["pre1900"]
    )
    if verified_table != (POINT_BUNDLE / "formulas.jsonl.gz"):
        raise RuntimeError("formula table path disagreement")
    model, run_spec = load_model(CHECKPOINT, graph_store, device)
    if int(run_spec["knowledge_cutoff"]) != 1899:
        raise RuntimeError("not a Phase-A pre-1900 checkpoint")

    memories, frozen_cdf, memory_audit = build_or_load_memory(
        model, graph_store, formula_to_identity, point_config, device,
        args.batch_size, args.graph_batch_size, args.workers, args.points, args.top_k,
    )
    if any(len(values) != 91 for values in frozen_cdf.values()):
        raise RuntimeError("calibration CDF does not contain the published 91 formula units")

    validation_ids, validation_cloud = score_curated_archive(
        model, memories, VALIDATION_ARCHIVE, args.batch_size, args.points, args.top_k, device
    )
    validation = add_frozen_ensemble(aggregate_by_key(validation_ids, validation_cloud), frozen_cdf)
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))["sets"]["curated_1901_1950"]["scores"]
    common = sorted(set(validation) & set(published))
    if len(common) != 38:
        raise RuntimeError(f"published validation family mismatch: {len(common)}")
    deltas = [abs(validation[key]["one_class_ensemble"] - published[key]["one_class_ensemble"]) for key in common]
    max_delta = float(max(deltas))
    if max_delta > 0.03:
        raise RuntimeError(f"reconstructed score drift exceeds tolerance: {max_delta}")

    family_ids, cloud_scores = score_curated_archive(
        model, memories, EXPANSION, args.batch_size, args.points, args.top_k, device
    )
    family_scores = add_frozen_ensemble(aggregate_by_key(family_ids, cloud_scores), frozen_cdf)
    registry = json.loads(EXPANSION_REGISTRY.read_text(encoding="utf-8"))
    meta = {row["family_id"]: row for row in registry["cases"]}
    rows = [
        {
            "family_id": key,
            "short_name": meta[key]["short_name"],
            "first_valid_year": meta[key]["first_valid_year"],
            **value,
        }
        for key, value in sorted(family_scores.items(), key=lambda item: meta[item[0]]["first_valid_year"])
    ]
    output = {
        "status": "PASS",
        "scope": "post-score case-study expansion; descriptive forward inference only",
        "no_training_audit": {
            "training_steps": 0, "optimizer_constructed": False,
            "weights_updated": False, "calibration_updated": False,
            "original_55_family_benchmark_mutated": False,
        },
        "provenance": {
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
            "expansion_archive_sha256": sha256(EXPANSION),
            "expansion_registry_sha256": sha256(EXPANSION_REGISTRY),
            "published_phase_a_result_sha256": sha256(PUBLISHED),
            "identity_groups": len(identity_names),
        },
        "memory_audit": memory_audit,
        "reconstruction_check": {
            "benchmark": "published curated 1901--1950 set",
            "families_compared": len(common),
            "maximum_absolute_ensemble_delta": max_delta,
            "tolerance": 0.03,
        },
        "case_rows": rows,
        "interpretation": "The large learned score is added as a frozen descriptive component; these post-selected families do not enter a revised AUROC.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"], "memory_audit": memory_audit,
        "reconstruction_check": output["reconstruction_check"],
        "scores": {row["short_name"]: row["one_class_ensemble"] for row in rows},
    }, indent=2))


if __name__ == "__main__":
    main()
