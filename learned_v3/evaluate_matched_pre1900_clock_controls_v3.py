"""Post-score generator-domain audit for a frozen EPOCH v3 checkpoint.

This reuses the public functions of the hash-pinned evaluator to construct the
same training memories and calibration CDF, then scores a separately frozen
archive containing canonical representatives of all nine Phase-A internal
families.  No model or score parameter is fitted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from epoch_formula_graph_dataset import EpochFormulaGraphStore
from evaluate_epoch_horizon_v3 import (
    add_frozen_ensemble,
    aggregate_by_key,
    clustered_bootstrap_auc,
    compact_rows,
    fit_empirical_cdf,
    graph_memories,
    load_model,
    metric_values,
    point_formula_memory,
    read_formula_table,
    score_curated_archive,
    score_pointcloud_role,
    sha256,
    shrinkage_gaussian_memory,
)
from train_epoch_horizon_v3 import contrastive_identity_lookup


METRIC = "one_class_ensemble"


def result_scores(result: dict, set_name: str) -> dict[str, dict[str, float]]:
    return {
        str(key): {str(name): float(value) for name, value in row.items()}
        for key, row in result["sets"][set_name]["scores"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--graph-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal matched-control audit")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    primary_path = args.primary_result.resolve()
    archive_path = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(config["knowledge_cutoff"]) != 1899 or int(primary["knowledge_cutoff"]) != 1899:
        raise RuntimeError("matched pre-1900 domain audit is defined for Phase A only")
    if primary["status"] != "PASS" or manifest["status"] != "PASS":
        raise RuntimeError("input evidence is not frozen PASS")
    if primary["provenance"]["checkpoint_sha256"] != sha256(checkpoint_path):
        raise RuntimeError("primary result/checkpoint disagreement")
    if primary["provenance"]["config_sha256"] != sha256(config_path):
        raise RuntimeError("primary result/config disagreement")
    expected_archive = manifest["assets"][archive_path.name]["sha256"]
    if sha256(archive_path) != expected_archive:
        raise RuntimeError("matched-control archive hash mismatch")

    point_config = config["pointcloud"]
    provenance_mode = point_config["headline_provenance_mode"]
    known_horizons = ["pre1900"]
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    graph_store = EpochFormulaGraphStore(config["formula_graph"]["bundle"])
    _formula_rows, formula_table_path = read_formula_table(point_config["bundle"])
    formula_to_identity, _identity_names, verified_table = contrastive_identity_lookup(
        graph_store, point_config["bundle"], known_horizons
    )
    if verified_table != formula_table_path:
        raise RuntimeError("formula table path disagreement")
    model, run_spec = load_model(checkpoint_path, graph_store, device)
    if int(run_spec["knowledge_cutoff"]) != 1899:
        raise RuntimeError("checkpoint is not the Phase-A horizon")

    memories = graph_memories(
        model, graph_store, formula_to_identity, point_config["view"],
        provenance_mode, known_horizons, device, args.graph_batch_size,
    )
    point_memory, point_memory_indices = point_formula_memory(
        model, point_config, provenance_mode, known_horizons,
        args.batch_size, args.points, args.workers, len(graph_store), device,
    )
    memories["point_formula"] = point_memory
    memories["point_formula_indices"] = point_memory_indices
    memories["point_gaussian"] = shrinkage_gaussian_memory(point_memory)

    calibration_ids, calibration_cloud_scores = score_pointcloud_role(
        model, memories, point_config, "calibration", provenance_mode,
        known_horizons, args.batch_size, args.points, args.workers,
        args.top_k, device,
    )
    calibration_formula = aggregate_by_key(calibration_ids, calibration_cloud_scores)
    frozen_cdf = fit_empirical_cdf(calibration_formula)
    family_ids, control_cloud_scores = score_curated_archive(
        model, memories, archive_path, args.batch_size, args.points,
        args.top_k, device,
    )
    controls = add_frozen_ensemble(
        aggregate_by_key(family_ids, control_cloud_scores), frozen_cdf
    )

    native = result_scores(primary, "known_internal")
    expected_families = set(native)
    if set(controls) != expected_families:
        raise RuntimeError({
            "control_only": sorted(set(controls) - expected_families),
            "native_only": sorted(expected_families - set(controls)),
        })
    positive_sets = {
        name: result_scores(primary, name)
        for name in (
            "curated_1901_1950", "curated_post1950", "curated_all_post1900"
        )
    }
    comparisons = {
        f"matched_pre1900_vs_{name}": clustered_bootstrap_auc(
            metric_values(controls, METRIC), metric_values(rows, METRIC),
            args.bootstrap, args.seed + offset,
        )
        for offset, (name, rows) in enumerate(positive_sets.items())
    }
    comparisons["native_internal_vs_matched_pre1900_domain_shift"] = clustered_bootstrap_auc(
        metric_values(native, METRIC), metric_values(controls, METRIC),
        args.bootstrap, args.seed + 10,
    )

    score_names = sorted(next(iter(controls.values())))
    component_aurocs = {
        name: {
            score_name: float(
                clustered_bootstrap_auc(
                    metric_values(controls, score_name), metric_values(rows, score_name),
                    1, args.seed,
                )["auroc"]
            )
            for score_name in score_names
        }
        for name, rows in positive_sets.items()
    }
    result = {
        "status": "PASS",
        "schema_version": "3.0-post-score-domain-audit",
        "interpretation": "All nine frozen internal pre-1900 families regenerated with the curated post-1900 sampling/noise recipe; post-score diagnostic, not a new primary endpoint.",
        "knowledge_cutoff": 1899,
        "primary_metric_reused": METRIC,
        "future_horizon_used_for_training_or_score_fitting": False,
        "sets": {
            "native_known_internal": compact_rows(native),
            "generator_matched_pre1900_controls": compact_rows(controls),
        },
        "comparisons": comparisons,
        "component_aurocs_descriptive": component_aurocs,
        "provenance": {
            "config_sha256": sha256(config_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "primary_result_sha256": sha256(primary_path),
            "archive_sha256": sha256(archive_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "limitations": [
            "The diagnostic was designed after Phase-A temporal AUROCs were observed.",
            "Each family is represented by one canonical reduction rather than the full within-family formula distribution.",
            "Nine negative families still provide limited precision for specificity.",
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
