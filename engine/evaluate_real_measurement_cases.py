"""Score the frozen real-measurement case set with the frozen EPOCH components.

Forward inference only.  No weights, reference embeddings, calibration families,
decision thresholds or combination rules are fitted, refitted or reweighted
here, and no existing result file is modified.

This audit reports raw scores.  Every measurement view of every series is
released individually, together with the per-series and per-mechanism-group
medians and the threshold-free ranking statistic.  No alert threshold is applied
and no detection or false-positive count is formed: the released numbers are the
continuous component outputs, and any operating point is left to the reader.

The learned component uses the zero-unit encoder, its reference bank rebuilt
deterministically from the frozen pre-1900 training split.  Real series are
passed through the rescale-only view function, so no synthetic noise is added to
a measurement.  The declared near-constant routing rule is applied unchanged.

The statistical component reruns the complete four-form direction, window,
model-selection and scoring procedure.  Each series is subsampled to the largest
frozen calibration size it supports (n = 5, 9, 43 or 120), so scores are
size-comparable across series.

Inferential units are mechanism groups, never individual files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from evaluate_pre1900_cross_family import CHECKPOINT, VIEWS, real_views  # noqa: E402
from evaluate_pre1900_strict_encoder import DATA, encode, knn_scores, load_encoder  # noqa: E402
from evaluate_frozen_post1900_formulas import build_reference  # noqa: E402
from strict_evidence_audit import raw_score  # noqa: E402

ARCHIVE = ROOT / "data" / "real_measurement_case_pointclouds.npz"
REGISTRY = ROOT / "data" / "real_measurement_case_registry.json"
LEARNED_FREEZE = ROOT / "results" / "pre1900_cross_family_conformal.json"
STAT_FREEZE = ROOT / "results" / "calibrated_detector_v2.json"
OUTPUT = ROOT / "results" / "real_measurement_case_evaluation.json"
VIEW_TABLE = ROOT / "results" / "real_measurement_case_view_scores.csv"

INFERENCE_SEED = 20260912
BOOTSTRAP = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_size(available: int, sizes: list[int]) -> int | None:
    usable = [size for size in sizes if size <= available]
    return max(usable) if usable else None


def statistical_views(cloud: np.ndarray, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    views = []
    for _ in range(VIEWS):
        index = np.sort(rng.choice(len(cloud), size, replace=False))
        views.append(np.asarray(cloud, float)[index])
    return views


def grouped_bootstrap_auroc(
    positive: dict[str, float], negative: dict[str, float], seed: int
) -> tuple[float, list[float]]:
    """Threshold-free ranking statistic, resampled over mechanism groups."""
    pos = np.asarray(list(positive.values()), float)
    neg = np.asarray(list(negative.values()), float)
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    point = float(roc_auc_score(labels, np.r_[pos, neg]))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        if len(set(p)) == 1 and len(set(n)) == 1 and p[0] == n[0]:
            continue
        draws.append(roc_auc_score(labels, np.r_[p, n]))
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    learned_freeze = json.loads(LEARNED_FREEZE.read_text(encoding="utf-8"))
    stat_freeze = json.loads(STAT_FREEZE.read_text(encoding="utf-8"))
    archive = np.load(ARCHIVE, allow_pickle=False)

    assert sha256(CHECKPOINT) == learned_freeze["checkpoint_sha256"], "learned checkpoint changed"
    assert sha256(DATA) == learned_freeze["data_sha256"], "pre-1900 reference data changed"
    stat_sizes = sorted(int(k) for k in stat_freeze["calibration"])

    model, checkpoint = load_encoder(CHECKPOINT)
    assert checkpoint["classical_only"] is True, "checkpoint is not the classical-only encoder"
    assert checkpoint["uses_dimension_metadata"] is False, "checkpoint is not the zero-unit encoder"
    reference = build_reference(model, np.load(DATA, allow_pickle=True))

    meta = {row["dataset"]: row for row in registry["cases"]}
    ids = [str(value) for value in archive["dataset_ids"]]

    view_rows, rows = [], []
    for position, dataset in enumerate(ids):
        cloud = np.asarray(archive[f"cloud_{position}"], float)
        info = meta[dataset]

        # Declared routing rule, applied exactly as frozen in the cross-family audit.
        relative_variation = float(np.std(cloud[:, 1]) / (abs(np.mean(cloud[:, 1])) + 1e-12))
        if relative_variation < 0.001:
            learned_route = "abstain_from_generic_learned_component"
            learned_view_scores: list[float] = []
        else:
            learned_route = "generic_learned_component"
            rng = np.random.default_rng(INFERENCE_SEED + position)
            learned_view_scores = [
                float(value) for value in
                knn_scores(reference, encode(model, real_views(cloud, rng)))
            ]

        size = matched_size(len(cloud), stat_sizes)
        stat_view_scores: list[float] = []
        if size is not None:
            rng_stat = np.random.default_rng(INFERENCE_SEED + 10_000 + position)
            for view in statistical_views(cloud, size, rng_stat):
                value = raw_score(view[:, 0], view[:, 1], combiner="uncapped_log")
                stat_view_scores.append(
                    float(value) if value is not None and np.isfinite(value) else float("nan")
                )

        for index in range(VIEWS):
            view_rows.append({
                "dataset": dataset,
                "mechanism_group": info["mechanism_group"],
                "horizon_class": info["horizon_class"],
                "view": index,
                "learned_score": (learned_view_scores[index]
                                  if index < len(learned_view_scores) else ""),
                "statistical_matched_n": size if size is not None else "",
                "statistical_score": (stat_view_scores[index]
                                      if index < len(stat_view_scores) else ""),
            })

        finite_stat = [value for value in stat_view_scores if np.isfinite(value)]
        rows.append({
            "dataset": dataset,
            "mechanism_group": info["mechanism_group"],
            "horizon_class": info["horizon_class"],
            "points_used": info["points_used"],
            "incumbent_1899": info["incumbent_1899"],
            "relative_response_variation": relative_variation,
            "learned_route": learned_route,
            "learned_view_scores": learned_view_scores,
            "learned_score": (float(np.median(learned_view_scores))
                              if learned_view_scores else float("nan")),
            "statistical_matched_n": size,
            "statistical_view_scores": stat_view_scores,
            "statistical_score": (float(np.median(finite_stat))
                                  if finite_stat else float("nan")),
            "statistical_view_failures": int(len(stat_view_scores) - len(finite_stat)),
        })

    groups = sorted({row["mechanism_group"] for row in rows})
    group_rows = {}
    for group in groups:
        members = [row for row in rows if row["mechanism_group"] == group]
        learned = np.asarray([row["learned_score"] for row in members], float)
        learned = learned[np.isfinite(learned)]
        stat = np.asarray([row["statistical_score"] for row in members], float)
        stat = stat[np.isfinite(stat)]
        group_rows[group] = {
            "horizon_class": members[0]["horizon_class"],
            "datasets": [row["dataset"] for row in members],
            "learned_group_score": float(np.median(learned)) if len(learned) else float("nan"),
            "statistical_group_score": float(np.median(stat)) if len(stat) else float("nan"),
        }

    def split(key: str) -> tuple[dict, dict]:
        positive = {g: r[key] for g, r in group_rows.items()
                    if r["horizon_class"] == "post_1900_mechanism" and np.isfinite(r[key])}
        negative = {g: r[key] for g, r in group_rows.items()
                    if r["horizon_class"] == "classical_or_metrology" and np.isfinite(r[key])}
        return positive, negative

    learned_pos, learned_neg = split("learned_group_score")
    stat_pos, stat_neg = split("statistical_group_score")
    learned_auroc, learned_ci = grouped_bootstrap_auroc(learned_pos, learned_neg, 20260912)
    stat_auroc, stat_ci = grouped_bootstrap_auroc(stat_pos, stat_neg, 20260913)

    with VIEW_TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(view_rows[0]))
        writer.writeheader()
        writer.writerows(view_rows)

    output = {
        "status": "frozen forward-inference audit on unmodified NIST measurement series",
        "reporting_scope": (
            "raw component scores only; no alert threshold is applied and no detection or "
            "false-positive count is formed from these series"
        ),
        "no_training_audit": {
            "training_steps": 0, "weights_updated": False, "thresholds_updated": False,
            "fusion_weight_fitted": False, "existing_results_mutated": False,
        },
        "provenance": {
            "registry_sha256": sha256(REGISTRY), "archive_sha256": sha256(ARCHIVE),
            "encoder_sha256": sha256(CHECKPOINT), "pre1900_reference_data_sha256": sha256(DATA),
            "view_score_table": VIEW_TABLE.name,
        },
        "design": {
            "datasets": len(rows),
            "measurement_views_per_series": VIEWS,
            "released_view_scores": len(view_rows),
            "mechanism_groups": len(groups),
            "post_1900_mechanism_groups": len(learned_pos),
            "classical_or_metrology_groups": len(learned_neg),
            "inferential_unit": "mechanism group",
            "label_basis": registry["label_basis"],
            "learned_view_function": "rescale-only real_views; no synthetic noise added to a measurement",
            "statistical_size_matching": "each series subsampled to the largest frozen calibration size it supports",
        },
        "ranking": {
            "statistic": "grouped AUROC, threshold-free, resampled over mechanism groups",
            "learned_group_auroc": learned_auroc,
            "learned_group_auroc_95ci": learned_ci,
            "statistical_group_auroc": stat_auroc,
            "statistical_group_auroc_95ci": stat_ci,
        },
        "group_scores": group_rows,
        "series_scores": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"released {len(view_rows)} view scores over {len(rows)} series "
          f"and {len(groups)} mechanism groups")
    print(f"learned      group AUROC {learned_auroc:.3f} "
          f"[{learned_ci[0]:.3f}, {learned_ci[1]:.3f}]")
    print(f"statistical  group AUROC {stat_auroc:.3f} "
          f"[{stat_ci[0]:.3f}, {stat_ci[1]:.3f}]")
    print()
    order = sorted(group_rows, key=lambda g: -group_rows[g]["learned_group_score"])
    for group in order:
        row = group_rows[group]
        tag = "post-1900" if row["horizon_class"] == "post_1900_mechanism" else "classical "
        print(f"  {tag}  {group:34s} learned {row['learned_group_score']:.4f}   "
              f"statistical {row['statistical_group_score']:9.3f}")


if __name__ == "__main__":
    main()
