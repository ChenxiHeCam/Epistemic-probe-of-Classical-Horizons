"""Score the post-score case-study expansion with frozen EPOCH components.

This script performs forward inference only.  It does not modify the original
55-family benchmark, train either component, fit a fusion weight, or report a
new primary AUROC.  The successor/old-model residual drop is a simulator sanity
check because these records were generated from the successor reductions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from evaluate_frozen_post1900_formulas import (  # noqa: E402
    build_reference,
    grouped_median,
    learned_record_scores,
    stat_record_scores,
)
from evaluate_pre1900_cross_family import CHECKPOINT, DATA  # noqa: E402
from evaluate_pre1900_strict_encoder import load_encoder  # noqa: E402
from calibrated_detector_v2 import _calibration_task  # noqa: E402


ARCHIVE = ROOT / "data" / "case_study_expansion_pointclouds.npz"
REGISTRY = ROOT / "data" / "case_study_expansion_registry.json"
MANIFEST = ROOT / "results" / "case_study_expansion_manifest.json"
LEARNED_FREEZE = ROOT / "results" / "pre1900_cross_family_conformal.json"
ORIGINAL_EVAL = ROOT / "results" / "frozen_post1900_formula_evaluation.json"
OUTPUT = ROOT / "results" / "case_study_expansion_evaluation.json"
STAT_CAL_SOURCE = ROOT / "data" / "classical_pointclouds.npz"
STAT_CAL_FREEZE = ROOT / "results" / "calibrated_detector_v2.json"
STAT_CAL_CACHE = ROOT / "data" / "statistical_n120_calibration_scores.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empirical_percentile(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, float)
    reference = reference[np.isfinite(reference)]
    return float((np.sum(reference <= value) + 0.5) / (len(reference) + 1.0))


def residual_drop(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    observed = np.asarray(archive["X"], float)[:, :, 1]
    successor = np.asarray(archive["clean_y"], float)
    incumbent = np.asarray(archive["incumbent_y"], float)
    new_rmse = np.sqrt(np.mean((observed - successor) ** 2, axis=1))
    old_rmse = np.sqrt(np.mean((observed - incumbent) ** 2, axis=1))
    scale = np.maximum(np.std(successor, axis=1), 1e-10)
    floor = 1e-6 * scale
    return np.log10((old_rmse + floor) / (new_rmse + floor))


def reconstruct_stat_calibration(jobs: int = 8) -> np.ndarray:
    """Recreate only the already specified n=120 calibration block."""
    freeze = json.loads(STAT_CAL_FREEZE.read_text(encoding="utf-8"))
    source_sha = sha256(STAT_CAL_SOURCE)
    if STAT_CAL_CACHE.exists():
        cached = np.load(STAT_CAL_CACHE, allow_pickle=False)
        if str(cached["source_sha256"]) == source_sha and int(cached["seed"]) == 20260904:
            return np.asarray(cached["scores"], float)
    clouds = np.load(STAT_CAL_SOURCE, allow_pickle=False)["X"]
    seed = 20260904
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(clouds), 800, replace=False)
    tasks = [
        (clouds[index], 120, seed + 120 * 100_000 + position)
        for position, index in enumerate(indices)
    ]
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        values = list(pool.map(_calibration_task, tasks, chunksize=4))
    scores = np.asarray([value for value in values if value is not None and np.isfinite(value)], float)
    expected = float(freeze["synthetic_mechanism_benchmark"]["thresholds"]["5pct"])
    actual = float(np.quantile(scores, 0.95))
    if not np.isclose(actual, expected, rtol=0, atol=1e-10):
        raise RuntimeError(f"reconstructed n=120 q95 mismatch: {actual} != {expected}")
    np.savez_compressed(STAT_CAL_CACHE, scores=scores, source_sha256=np.asarray(source_sha), seed=np.asarray(seed))
    return scores


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry_payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_EVAL.read_text(encoding="utf-8"))
    learned_freeze = json.loads(LEARNED_FREEZE.read_text(encoding="utf-8"))
    archive = np.load(ARCHIVE, allow_pickle=False)
    assert manifest["status"] == "PASS"
    assert manifest["files"][ARCHIVE.name]["sha256"] == sha256(ARCHIVE)
    assert manifest["files"][REGISTRY.name]["sha256"] == sha256(REGISTRY)
    assert len(archive["X"]) == manifest["records"]
    assert sha256(CHECKPOINT) == original["frozen_artifacts"]["encoder_sha256"]
    assert sha256(DATA) == original["frozen_artifacts"]["pre1900_reference_data_sha256"]

    model, checkpoint = load_encoder(CHECKPOINT)
    assert checkpoint["classical_only"] is True
    assert checkpoint["uses_dimension_metadata"] is False
    reference = build_reference(model, np.load(DATA, allow_pickle=True))
    clouds = np.asarray(archive["X"], np.float32)
    families = np.asarray(archive["family_ids"]).astype(str)
    learned_record = learned_record_scores(model, reference, clouds)
    stat_record = stat_record_scores(clouds, max(1, min(8, len(clouds))))
    drop_record = residual_drop(archive)
    stat_calibration = reconstruct_stat_calibration()

    learned_family = grouped_median(learned_record, families)
    stat_family = grouped_median(stat_record, families)
    drop_family = grouped_median(drop_record, families)
    controls = original["heldout_pre1900_controls"]
    learned_control = np.asarray([row["learned_score"] for row in controls], float)
    stat_control = np.asarray([row["stat_score"] for row in controls], float)
    metadata = {row["family_id"]: row for row in registry_payload["cases"]}
    learned_calibration = np.asarray([
        row["median_score"] for row in learned_freeze["calibration"].values()
    ], float)

    rows = []
    for family in sorted(learned_family, key=lambda key: metadata[key]["first_valid_year"]):
        learned_score = float(learned_family[family])
        stat_score = float(stat_family[family])
        rows.append({
            "family_id": family,
            "short_name": metadata[family]["short_name"],
            "first_valid_year": metadata[family]["first_valid_year"],
            "shape_class": metadata[family]["shape_class"],
            "eligible_components": metadata[family]["eligible_components"],
            "learned_compact_score": learned_score,
            "learned_percentile_vs_11_heldout_pre1900": empirical_percentile(learned_score, learned_control),
            "learned_percentile_vs_40_training_family_calibration": empirical_percentile(learned_score, learned_calibration),
            "statistical_score": stat_score,
            "statistical_percentile_vs_11_heldout_pre1900": empirical_percentile(stat_score, stat_control),
            "statistical_percentile_vs_785_size_matched_calibration": empirical_percentile(stat_score, stat_calibration),
            "theory_clock_residual_drop_log10": float(drop_family[family]),
            "statistical_record_abstentions": int(np.sum(~np.isfinite(stat_record[families == family]))),
            "interpretive_note": metadata[family]["reduction_status"],
        })

    output = {
        "status": "completed frozen forward-inference development audit",
        "selection_timing": registry_payload["selection_timing"],
        "no_training_audit": {
            "training_steps": 0, "weights_updated": False,
            "thresholds_updated": False, "fusion_weight_fitted": False,
            "original_55_family_result_mutated": False,
        },
        "provenance": {
            "registry_sha256": sha256(REGISTRY), "archive_sha256": sha256(ARCHIVE),
            "compact_encoder_sha256": sha256(CHECKPOINT),
            "original_55_family_evaluation_sha256": sha256(ORIGINAL_EVAL),
            "statistical_n120_calibration_sha256": sha256(STAT_CAL_CACHE),
        },
        "benchmark": {
            "families": len(rows), "records": len(clouds),
            "records_per_family": manifest["records_per_family"],
            "points_per_record": manifest["points_per_record"],
            "negative_reference": "the same 11 cited held-out pre-1900 families as the frozen 55-family audit",
            "statistical_calibration_scores": len(stat_calibration),
        },
        "case_rows": rows,
        "theory_only_not_forced_through_pointcloud_encoder": registry_payload["theory_only_not_encoded_as_pointclouds"],
        "interpretation": {
            "primary_status": "descriptive post-score expansion; not a new AUROC denominator",
            "learned_and_statistical": "both components were run on every eligible point cloud without retraining or score fusion",
            "theory_clock_residual_drop": "simulator sanity check only; the successor curve generated each record",
            "why_theory_only_abstains": "a scalar coefficient or sparse level splitting is not converted into an artificial 120-point curve merely to satisfy the learned input contract",
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "case_rows": rows}, indent=2))


if __name__ == "__main__":
    main()
