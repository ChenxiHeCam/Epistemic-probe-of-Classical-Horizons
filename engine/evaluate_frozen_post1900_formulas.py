"""Evaluate the frozen EPOCH components on post-1900 formula point clouds.

No weights, reference embeddings, calibration families, decision thresholds or
combination rules are optimized here.  The zero-unit encoder is loaded in eval
mode; its reference bank is reconstructed deterministically from the already
frozen pre-1900 training split.  Statistical scores use the already frozen
120-point q95/q99 thresholds from calibrated_detector_v2.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from evaluate_pre1900_cross_family import CHECKPOINT, DATA, SEED, classical_views
from evaluate_pre1900_strict_encoder import encode, knn_scores, load_encoder
from strict_evidence_audit import raw_score


POST_DATA = ROOT / "data" / "post1900_formula_pointclouds.npz"
POST_REGISTRY = ROOT / "data" / "post1900_formula_registry.json"
POST_MANIFEST = ROOT / "results" / "post1900_formula_benchmark_manifest.json"
LEARNED_FREEZE = ROOT / "results" / "pre1900_cross_family_conformal.json"
STAT_FREEZE = ROOT / "results" / "calibrated_detector_v2.json"
OUTPUT = ROOT / "results" / "frozen_post1900_formula_evaluation.json"
INFERENCE_SEED = 20260907


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_median(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    output = {}
    for group in sorted(set(groups.astype(str))):
        subset = values[groups == group]
        subset = subset[np.isfinite(subset)]
        output[group] = float(np.median(subset)) if len(subset) else float("nan")
    return output


def grouped_any(values: np.ndarray, groups: np.ndarray) -> dict[str, bool]:
    return {
        group: bool(np.any(values[groups == group]))
        for group in sorted(set(groups.astype(str)))
    }


def rate_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson score interval, used only as a descriptive family-rate interval."""
    if total == 0:
        return [float("nan"), float("nan")]
    p = hits / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total))/denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    lower = 0.0 if hits == 0 else max(0, centre-half)
    upper = 1.0 if hits == total else min(1, centre+half)
    return [float(lower), float(upper)]


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, seed: int, repetitions: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    positions = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    values = []
    for _ in range(repetitions):
        draw = np.concatenate([
            rng.choice(positions[label], len(positions[label]), replace=True)
            for label in (0, 1)
        ])
        values.append(roc_auc_score(labels[draw], scores[draw]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def build_reference(model, archive: np.lib.npyio.NpzFile) -> np.ndarray:
    """Reconstruct exactly the frozen reference-bank protocol; never update it."""
    split = np.asarray(archive["splits"]).astype(str)
    train_indices = np.flatnonzero(split == "train")
    rng = np.random.default_rng(SEED)
    views = []
    for index in train_indices:
        views.extend(classical_views(archive["X"][index], rng))
    return encode(model, views)


def learned_record_scores(model, reference: np.ndarray, clouds: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(INFERENCE_SEED)
    all_views = []
    for cloud in clouds:
        all_views.extend(classical_views(cloud, rng))
    scores = knn_scores(reference, encode(model, all_views))
    return np.median(scores.reshape(len(clouds), -1), axis=1)


def _stat_score(cloud: np.ndarray) -> float | None:
    return raw_score(cloud[:, 0], cloud[:, 1], combiner="uncapped_log")


def stat_record_scores(clouds: np.ndarray, jobs: int) -> np.ndarray:
    if jobs == 1:
        values = [_stat_score(cloud) for cloud in clouds]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            values = list(pool.map(_stat_score, clouds, chunksize=4))
    return np.asarray([np.nan if value is None else float(value) for value in values])


def family_rows(
    registry: list[dict], family_ids: np.ndarray,
    learned_scores: np.ndarray, stat_scores: np.ndarray,
    learned_threshold: float, stat_q95: float, stat_q99: float,
    calibration_values: np.ndarray,
) -> list[dict]:
    learned = grouped_median(learned_scores, family_ids)
    stat = grouped_median(stat_scores, family_ids)
    stat_abstain = {
        family: int(np.sum(~np.isfinite(stat_scores[family_ids == family])))
        for family in sorted(set(family_ids.astype(str)))
    }
    metadata = {row["family_id"]: row for row in registry}
    rows = []
    for family in sorted(learned):
        row = metadata[family]
        learned_flag = learned[family] > learned_threshold
        stat_flag_5 = np.isfinite(stat[family]) and stat[family] > stat_q95
        stat_flag_1 = np.isfinite(stat[family]) and stat[family] > stat_q99
        learned_p = float((1 + np.sum(calibration_values >= learned[family])) / (len(calibration_values) + 1))
        rows.append({
            "family_id": family, "law_name": row["law_name"],
            "first_valid_year": row["first_valid_year"], "domain": row["domain"],
            "shape_class": row["shape_class"], "functional_signature": row["signature"],
            "learned_score": learned[family], "learned_family_conformal_p": learned_p,
            "learned_alert_q95": bool(learned_flag),
            "stat_score": stat[family], "stat_alert_q95": bool(stat_flag_5),
            "stat_alert_q99": bool(stat_flag_1), "stat_record_abstentions": stat_abstain[family],
            "framework_or_alert_q95": bool(learned_flag or stat_flag_5),
            "framework_and_alert_q95": bool(learned_flag and stat_flag_5),
        })
    return rows


def summarize(rows: list[dict], key: str, value: str) -> dict:
    result = {}
    for group in sorted(set(row[key] for row in rows)):
        subset = [row for row in rows if row[key] == group]
        hits = sum(bool(row[value]) for row in subset)
        result[str(group)] = {
            "families": len(subset), "alerts": hits, "rate": hits/len(subset),
            "wilson_95_ci": rate_interval(hits, len(subset)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads(POST_MANIFEST.read_text(encoding="utf-8"))
    registry_payload = json.loads(POST_REGISTRY.read_text(encoding="utf-8"))
    post = np.load(POST_DATA, allow_pickle=True)
    pre = np.load(DATA, allow_pickle=True)
    learned_freeze = json.loads(LEARNED_FREEZE.read_text(encoding="utf-8"))
    stat_freeze = json.loads(STAT_FREEZE.read_text(encoding="utf-8"))

    assert manifest["status"] == "PASS"
    assert manifest["families"] == len(registry_payload["laws"])
    assert manifest["records"] == len(post["X"])
    assert manifest["files"][POST_DATA.name]["sha256"] == sha256(POST_DATA)
    assert manifest["files"][POST_REGISTRY.name]["sha256"] == sha256(POST_REGISTRY)
    assert all(row["first_valid_year"] > 1900 for row in registry_payload["laws"])

    model, checkpoint = load_encoder(CHECKPOINT)
    assert checkpoint["classical_only"] is True
    assert checkpoint["uses_dimension_metadata"] is False
    assert sha256(CHECKPOINT) == learned_freeze["checkpoint_sha256"]
    assert sha256(DATA) == learned_freeze["data_sha256"]
    model.eval()
    reference = build_reference(model, pre)

    learned_threshold = float(learned_freeze["protocol"]["threshold"])
    calibration_values = np.asarray([
        row["median_score"] for row in learned_freeze["calibration"].values()
    ], float)
    stat_thresholds = stat_freeze["synthetic_mechanism_benchmark"]["thresholds"]
    stat_q95 = float(stat_thresholds["5pct"])
    stat_q99 = float(stat_thresholds["1pct"])
    assert stat_freeze["synthetic_mechanism_benchmark"]["sample_size"] == 120
    assert post["X"].shape[1] == 120

    post_families = np.asarray(post["family_ids"]).astype(str)
    post_learned_records = learned_record_scores(model, reference, post["X"])
    post_stat_records = stat_record_scores(post["X"], max(1, args.jobs))
    rows = family_rows(
        registry_payload["laws"], post_families, post_learned_records, post_stat_records,
        learned_threshold, stat_q95, stat_q99, calibration_values,
    )

    # Independent cited pre-1900 validation+test families are negative controls.
    heldout_mask = np.asarray(pre["splits"]).astype(str) != "train"
    heldout_clouds = np.asarray(pre["X"])[heldout_mask, :120]
    heldout_families = np.asarray(pre["family_ids"]).astype(str)[heldout_mask]
    heldout_stat_records = stat_record_scores(heldout_clouds, max(1, args.jobs))
    heldout_stat_family = grouped_median(heldout_stat_records, heldout_families)
    heldout_learned_family = {}
    for split_name in ("validation", "test"):
        heldout_learned_family.update(
            learned_freeze["heldout_classical"][split_name]["family_scores"]
        )
    assert set(heldout_stat_family) == set(heldout_learned_family)

    control_rows = []
    for family in sorted(heldout_learned_family):
        learned_score = float(heldout_learned_family[family])
        stat_score = float(heldout_stat_family[family])
        learned_flag = learned_score > learned_threshold
        stat_flag = stat_score > stat_q95
        control_rows.append({
            "family_id": family, "learned_score": learned_score,
            "learned_alert_q95": learned_flag, "stat_score": stat_score,
            "stat_alert_q95": stat_flag,
            "framework_or_alert_q95": bool(learned_flag or stat_flag),
            "framework_and_alert_q95": bool(learned_flag and stat_flag),
        })

    post_learned_family = np.asarray([row["learned_score"] for row in rows])
    post_stat_family = np.asarray([row["stat_score"] for row in rows])
    control_learned_family = np.asarray([row["learned_score"] for row in control_rows])
    control_stat_family = np.asarray([row["stat_score"] for row in control_rows])
    learned_labels = np.r_[np.zeros(len(control_rows), int), np.ones(len(rows), int)]
    learned_values = np.r_[control_learned_family, post_learned_family]
    stat_valid = np.isfinite(np.r_[control_stat_family, post_stat_family])
    stat_labels = learned_labels[stat_valid]
    stat_values = np.r_[control_stat_family, post_stat_family][stat_valid]

    def endpoint(alert_key: str) -> dict:
        post_hits = sum(row[alert_key] for row in rows)
        control_hits = sum(row[alert_key] for row in control_rows)
        return {
            "post1900_alerts": post_hits, "post1900_families": len(rows),
            "post1900_rate": post_hits/len(rows),
            "post1900_wilson_95_ci": rate_interval(post_hits, len(rows)),
            "heldout_pre1900_false_alerts": control_hits,
            "heldout_pre1900_families": len(control_rows),
            "heldout_pre1900_fpr": control_hits/len(control_rows),
            "heldout_pre1900_wilson_95_ci": rate_interval(control_hits, len(control_rows)),
        }

    signature_rows = []
    for signature in sorted(set(row["functional_signature"] for row in rows)):
        subset = [row for row in rows if row["functional_signature"] == signature]
        signature_rows.append({
            "functional_signature": signature, "source_law_families": len(subset),
            "learned_alert": bool(np.median([row["learned_score"] for row in subset]) > learned_threshold),
            "stat_alert": bool(np.median([row["stat_score"] for row in subset]) > stat_q95),
        })

    chronology = {}
    for label, lower, upper in (("1901-1919", 1901, 1919), ("1920-1949", 1920, 1949), ("1950-1999", 1950, 1999)):
        subset = [row for row in rows if lower <= row["first_valid_year"] <= upper]
        chronology[label] = {
            "families": len(subset),
            "learned_alert_rate": float(np.mean([row["learned_alert_q95"] for row in subset])),
            "stat_alert_rate": float(np.mean([row["stat_alert_q95"] for row in subset])),
            "framework_or_alert_rate": float(np.mean([row["framework_or_alert_q95"] for row in subset])),
        }

    result = {
        "status": "completed frozen forward-inference audit",
        "no_training_audit": {
            "training_steps": 0, "optimizer_constructed": False, "model_eval_mode": not model.training,
            "weights_updated": False, "thresholds_updated": False,
            "post1900_records_entered_training_or_calibration": False,
        },
        "frozen_artifacts": {
            "encoder": CHECKPOINT.name, "encoder_sha256": sha256(CHECKPOINT),
            "pre1900_reference_data": DATA.name, "pre1900_reference_data_sha256": sha256(DATA),
            "learned_threshold_q95_family": learned_threshold,
            "learned_calibration_families": len(calibration_values),
            "learned_minimum_resolvable_conformal_p": 1/(len(calibration_values)+1),
            "stat_threshold_q95_n120": stat_q95, "stat_threshold_q99_n120": stat_q99,
            "learned_protocol_source": LEARNED_FREEZE.name,
            "stat_protocol_source": STAT_FREEZE.name,
        },
        "benchmark": {
            "registry": POST_REGISTRY.name, "registry_sha256": sha256(POST_REGISTRY),
            "pointcloud_archive": POST_DATA.name, "pointcloud_archive_sha256": sha256(POST_DATA),
            "source_law_families": len(rows), "records": len(post["X"]),
            "records_per_family": manifest["records_per_family"], "points_per_record": post["X"].shape[1],
            "strict_year_range": [manifest["year_min"], manifest["year_max"]],
            "negative_controls": "11 cited source-law families held out from pre-1900 encoder training",
        },
        "primary_family_level": {
            "learned": {
                **endpoint("learned_alert_q95"),
                "auroc_post1900_vs_heldout_pre1900": float(roc_auc_score(learned_labels, learned_values)),
                "family_bootstrap_95_ci": bootstrap_auc_ci(learned_labels, learned_values, INFERENCE_SEED),
            },
            "statistical_four_form": {
                **endpoint("stat_alert_q95"),
                "q99_post1900_alerts": sum(row["stat_alert_q99"] for row in rows),
                "q99_post1900_rate": float(np.mean([row["stat_alert_q99"] for row in rows])),
                "auroc_post1900_vs_heldout_pre1900": float(roc_auc_score(stat_labels, stat_values)),
                "family_bootstrap_95_ci": bootstrap_auc_ci(stat_labels, stat_values, INFERENCE_SEED+1),
                "post1900_record_abstentions": int(np.sum(~np.isfinite(post_stat_records))),
                "heldout_pre1900_record_abstentions": int(np.sum(~np.isfinite(heldout_stat_records))),
            },
            "framework_or": endpoint("framework_or_alert_q95"),
            "framework_and": endpoint("framework_and_alert_q95"),
        },
        "by_shape_class": {
            method: summarize(rows, "shape_class", method)
            for method in ("learned_alert_q95", "stat_alert_q95", "framework_or_alert_q95", "framework_and_alert_q95")
        },
        "by_chronology_descriptive_only": chronology,
        "unique_functional_signatures": {
            "count": len(signature_rows),
            "learned_alerts": sum(row["learned_alert"] for row in signature_rows),
            "stat_alerts": sum(row["stat_alert"] for row in signature_rows),
            "rows": signature_rows,
        },
        "heldout_pre1900_controls": control_rows,
        "post1900_families": rows,
        "interpretation": {
            "estimand": "discrimination and alert rate for later formula shapes relative to a frozen pre-1900 representation/four-form extrapolation task",
            "not_estimand": "real-measurement discovery rate, historical publication-year classification, or rejection of all pre-1900 physics",
            "expected_hard_cases": "Post-1900 laws exactly expressible as affine, power or exponential curves can correctly remain unflagged by a shape-only system.",
            "multiple_testing": "Per-family learned conformal p-values are descriptive; with 40 calibration families their minimum is 1/41, so they cannot support a 55-test BH claim.",
            "framework_combination": "OR and AND rows are transparent component combinations at their existing q95 thresholds; no new ensemble weight or threshold was fitted.",
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "benchmark": result["benchmark"],
        "primary_family_level": result["primary_family_level"],
        "by_shape_class": result["by_shape_class"],
    }, indent=2))


if __name__ == "__main__":
    main()
