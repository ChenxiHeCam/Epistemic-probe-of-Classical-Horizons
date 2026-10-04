"""Leakage-safe one-class evaluation for a frozen EPOCH learned checkpoint.

The query path consumes only normalized unordered point clouds.  Formula ASTs
are encoded once to form the frozen memory of the training horizon; no future
formula or breakdown/deformation label is used to fit the detector.  Scores are
aggregated cloud -> formula -> source-law family/relation group before AUROC.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from epoch_formula_graph_dataset import (  # noqa: E402
    EpochFormulaGraphStore,
    PROVENANCE_MODES,
)
from epoch_learned_v3 import EpochArchitecture, EpochDualEncoder  # noqa: E402
from epoch_pointcloud_dataset import make_loader  # noqa: E402
from train_epoch_horizon_v3 import (  # noqa: E402
    contrastive_identity_lookup,
    to_device_graph,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_formula_table(pointcloud_bundle: str | Path) -> tuple[list[dict], Path]:
    path = Path(pointcloud_bundle).resolve() / "formulas.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()], path


def robust_normalize_clouds(clouds: np.ndarray) -> np.ndarray:
    """Apply the same per-cloud location/scale convention as generation."""
    clouds = np.asarray(clouds, dtype=np.float32)
    location = np.median(clouds, axis=1, keepdims=True)
    q25 = np.quantile(clouds, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(clouds, 0.75, axis=1, keepdims=True)
    scale = q75 - q25
    fallback = np.std(clouds, axis=1, keepdims=True)
    threshold = 1e-9 * (np.abs(location) + 1.0)
    scale = np.where(np.isfinite(scale) & (scale > threshold), scale, fallback)
    scale = np.maximum(scale, 1e-8)
    normalized = (clouds - location) / scale
    if not np.isfinite(normalized).all():
        raise RuntimeError("non-finite values after curated-cloud normalization")
    return normalized.astype(np.float32)


def deterministic_subsample(
    clouds: np.ndarray, record_ids: np.ndarray, points: int
) -> np.ndarray:
    if clouds.shape[1] == points:
        return clouds
    selected = np.empty((len(clouds), points, 2), dtype=np.float32)
    for index, (cloud, record_id) in enumerate(zip(clouds, record_ids)):
        seed = int.from_bytes(
            hashlib.sha256(str(record_id).encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        choice = np.sort(rng.choice(len(cloud), points, replace=len(cloud) < points))
        selected[index] = cloud[choice]
    return selected


def load_model(
    checkpoint_path: Path,
    graph_store: EpochFormulaGraphStore,
    device: torch.device,
) -> tuple[EpochDualEncoder, dict]:
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=False
    )
    run_spec = checkpoint["run_spec"]
    architecture = EpochArchitecture(**run_spec["architecture"])
    model = EpochDualEncoder(
        graph_store.manifest["node_vocab_size"], architecture
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, run_spec


@torch.inference_mode()
def graph_memories(
    model: EpochDualEncoder,
    store: EpochFormulaGraphStore,
    formula_to_identity: torch.Tensor,
    view: str,
    provenance_mode: str,
    horizons: list[str],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    role_key = f"{view}_role"
    tiers = PROVENANCE_MODES[provenance_mode]
    indices = [
        int(row["formula_index"])
        for row in store.metadata
        if row[role_key] == "train"
        and row["tier"] in tiers
        and row["horizon"] in horizons
    ]
    if not indices:
        raise RuntimeError("empty training-horizon graph memory")
    embeddings = []
    for offset in range(0, len(indices), batch_size):
        graph = to_device_graph(
            store.batch(indices[offset:offset + batch_size], deduplicate=False),
            device,
        )
        # Keep structural teacher FP32, matching training.
        with torch.autocast("cuda", enabled=False):
            embeddings.append(model.formula_encoder(graph).float().cpu())
    formula_memory = F.normalize(torch.cat(embeddings), dim=-1).to(device)
    identity_ids = formula_to_identity[torch.tensor(indices, dtype=torch.long)]
    unique_identity, inverse = torch.unique(identity_ids, sorted=True, return_inverse=True)
    identity_sum = torch.zeros(
        len(unique_identity), formula_memory.shape[1], dtype=torch.float32,
        device=device,
    )
    identity_sum.index_add_(0, inverse.to(device), formula_memory)
    counts = torch.bincount(inverse, minlength=len(unique_identity)).to(
        device=device, dtype=torch.float32
    )
    identity_memory = F.normalize(identity_sum / counts[:, None], dim=-1)
    return {
        "graph_formula": formula_memory,
        "graph_identity": identity_memory,
        "formula_indices": torch.tensor(indices, dtype=torch.long),
        "identity_ids": unique_identity,
    }


@torch.inference_mode()
def point_formula_memory(
    model: EpochDualEncoder,
    point_config: dict,
    provenance_mode: str,
    horizons: list[str],
    batch_size: int,
    points: int,
    workers: int,
    formula_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average training-cloud embeddings per exact known formula."""
    _dataset, _sampler, loader = make_loader(
        point_config["bundle"], point_config["view_index"],
        point_config["view"], "train", provenance_mode, batch_size,
        points, workers, shuffle=False, seed=20260907, horizons=horizons,
    )
    dimension = model.architecture.embedding_dim
    sums = torch.zeros(formula_count, dimension, dtype=torch.float32, device=device)
    counts = torch.zeros(formula_count, dtype=torch.long, device=device)
    for batch in loader:
        cloud = batch["points"].to(device, non_blocking=True)
        formula_id = batch["formula_index"].to(device, non_blocking=True)
        mask = torch.ones(cloud.shape[:2], dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model.point_encoder(cloud, mask)
        sums.index_add_(0, formula_id, F.normalize(embedding.float(), dim=-1))
        counts.index_add_(0, formula_id, torch.ones_like(formula_id))
    # ``torch.flatnonzero`` is not part of the public PyTorch API (NumPy has
    # ``flatnonzero``).  Use the exactly equivalent public operation so the
    # frozen evaluator runs on the preregistered PyTorch 2.14 environment.
    formula_indices = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if not len(formula_indices):
        raise RuntimeError("empty empirical point-formula memory")
    prototypes = F.normalize(
        sums[formula_indices] / counts[formula_indices, None].float(), dim=-1
    )
    return prototypes, formula_indices.cpu()


def shrinkage_gaussian_memory(prototypes: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """Fit a deterministic, well-conditioned Gaussian to known prototypes.

    The shrinkage intensity p/(n+p) is fixed by dimensionality and sample size,
    not selected on future labels.  It approaches the empirical covariance for
    large reference banks and the spherical target for small banks.
    """
    values = prototypes.float()
    count, dimension = values.shape
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(1, count - 1)
    spherical_variance = (covariance.trace() / dimension).clamp_min(1e-8)
    shrinkage = float(dimension / (count + dimension))
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * spherical_variance * torch.eye(
            dimension, dtype=covariance.dtype, device=covariance.device
        )
    )
    precision = torch.linalg.inv(covariance)
    return {"mean": mean, "precision": precision, "shrinkage": shrinkage}


def raw_anomaly_scores(
    embedding: torch.Tensor,
    memories: dict[str, torch.Tensor],
    top_k: int,
) -> dict[str, np.ndarray]:
    query = F.normalize(embedding.float(), dim=-1)
    result = {}
    for name in ("graph_formula", "graph_identity", "point_formula"):
        memory = memories[name]
        k = min(top_k, len(memory))
        similarity = query @ memory.T
        nearest = similarity.topk(k, dim=1, largest=True, sorted=False).values
        result[f"{name}_knn_distance"] = (
            1.0 - nearest.mean(dim=1)
        ).float().cpu().numpy()
    gaussian = memories["point_gaussian"]
    delta = query - gaussian["mean"]
    result["point_shrinkage_mahalanobis"] = torch.einsum(
        "bi,ij,bj->b", delta, gaussian["precision"], delta
    ).float().cpu().numpy()
    return result


def append_batch_scores(
    target: dict[str, list[np.ndarray]], scores: dict[str, np.ndarray]
) -> None:
    for name, values in scores.items():
        target[name].append(values)


def merge_batch_scores(target: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(parts) for name, parts in target.items()}


@torch.inference_mode()
def score_pointcloud_role(
    model: EpochDualEncoder,
    memories: dict[str, torch.Tensor],
    point_config: dict,
    role: str,
    provenance_mode: str,
    horizons: list[str],
    batch_size: int,
    points: int,
    workers: int,
    top_k: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _dataset, _sampler, loader = make_loader(
        point_config["bundle"], point_config["view_index"],
        point_config["view"], role, provenance_mode, batch_size,
        points, workers, shuffle=False, seed=20260907, horizons=horizons,
    )
    formula_ids = []
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in loader:
        cloud = batch["points"].to(device, non_blocking=True)
        mask = torch.ones(cloud.shape[:2], dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model.point_encoder(cloud, mask)
        append_batch_scores(collected, raw_anomaly_scores(embedding, memories, top_k))
        formula_ids.append(batch["formula_index"].numpy())
    return np.concatenate(formula_ids), merge_batch_scores(collected)


@torch.inference_mode()
def score_curated_archive(
    model: EpochDualEncoder,
    memories: dict[str, torch.Tensor],
    path: Path,
    batch_size: int,
    points: int,
    top_k: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    archive = np.load(path.resolve(), allow_pickle=False)
    record_ids = archive["record_ids"]
    clouds = deterministic_subsample(
        robust_normalize_clouds(archive["X"]), record_ids, points
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(clouds)), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for (cloud,) in loader:
        cloud = cloud.to(device, non_blocking=True)
        mask = torch.ones(cloud.shape[:2], dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model.point_encoder(cloud, mask)
        append_batch_scores(collected, raw_anomaly_scores(embedding, memories, top_k))
    return archive["family_ids"].astype(str), merge_batch_scores(collected)


def aggregate_by_key(
    keys: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys.tolist()):
        grouped[str(key)].append(index)
    return {
        key: {
            name: float(np.median(values[indices]))
            for name, values in scores.items()
        }
        for key, indices in grouped.items()
    }


def formula_to_scientific_groups(
    formula_scores: dict[str, dict[str, float]], formula_rows: list[dict]
) -> dict[str, dict[str, float]]:
    accumulated: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for formula_key, score in formula_scores.items():
        row = formula_rows[int(formula_key)]
        groups = list(map(str, row.get("historical_families") or []))
        if not groups:
            groups = [str(row["leakage_group_id"])]
        for group in sorted(set(groups)):
            for name, value in score.items():
                accumulated[group][name].append(value)
    return {
        group: {
            name: float(np.median(values)) for name, values in metrics.items()
        }
        for group, metrics in accumulated.items()
    }


def fit_empirical_cdf(calibration: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
    metric_names = next(iter(calibration.values())).keys()
    return {
        name: np.sort(np.asarray([row[name] for row in calibration.values()]))
        for name in metric_names
    }


def add_frozen_ensemble(
    rows: dict[str, dict[str, float]], cdf: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    result = {}
    for key, row in rows.items():
        enriched = dict(row)
        percentiles = []
        for name, reference in cdf.items():
            percentile = (
                np.searchsorted(reference, row[name], side="right") + 1.0
            ) / (len(reference) + 1.0)
            enriched[f"calibrated_{name}"] = float(percentile)
            percentiles.append(percentile)
        # Fixed before future data are loaded: equal-weight mean of the graph
        # formula, graph identity and empirical point-formula memory distances.
        enriched["one_class_ensemble"] = float(np.mean(percentiles))
        result[key] = enriched
    return result


def auroc(negative: np.ndarray, positive: np.ndarray) -> float:
    negative = np.asarray(negative, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    combined = np.concatenate((negative, positive))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    start = 0
    while start < len(combined):
        stop = start + 1
        while stop < len(combined) and combined[order[stop]] == combined[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    n_negative = len(negative)
    n_positive = len(positive)
    u = ranks[n_negative:].sum() - n_positive * (n_positive + 1) / 2.0
    return float(u / (n_negative * n_positive))


def clustered_bootstrap_auc(
    negative: np.ndarray,
    positive: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    estimate = auroc(negative, positive)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_negative = negative[rng.integers(0, len(negative), len(negative))]
        sampled_positive = positive[rng.integers(0, len(positive), len(positive))]
        values[index] = auroc(sampled_negative, sampled_positive)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "auroc": estimate,
        "ci95": [float(low), float(high)],
        "negative_groups": len(negative),
        "positive_groups": len(positive),
        "bootstrap_repetitions": repetitions,
    }


def metric_values(rows: dict[str, dict[str, float]], metric: str) -> np.ndarray:
    return np.asarray([row[metric] for row in rows.values()], dtype=np.float64)


def compact_rows(rows: dict[str, dict[str, float]]) -> dict:
    metrics = sorted(next(iter(rows.values())).keys()) if rows else []
    return {
        "groups": len(rows),
        "score_summary": {
            metric: {
                "median": float(np.median(metric_values(rows, metric))),
                "q25": float(np.quantile(metric_values(rows, metric), 0.25)),
                "q75": float(np.quantile(metric_values(rows, metric), 0.75)),
            }
            for metric in metrics
        },
        "scores": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        raise RuntimeError("CUDA is required for the formal learned evaluation")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    point_config = config["pointcloud"]
    cutoff = int(config["knowledge_cutoff"])
    if cutoff not in (1899, 1950):
        raise RuntimeError(f"unsupported knowledge cutoff: {cutoff}")
    provenance_mode = point_config["headline_provenance_mode"]
    known_horizons = ["pre1900"] if cutoff == 1899 else ["pre1900", "pre1950"]
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    graph_store = EpochFormulaGraphStore(config["formula_graph"]["bundle"])
    formula_rows, formula_table_path = read_formula_table(point_config["bundle"])
    formula_to_identity, identity_names, verified_table = contrastive_identity_lookup(
        graph_store, point_config["bundle"], known_horizons
    )
    if verified_table != formula_table_path:
        raise RuntimeError("formula table path disagreement")
    model, run_spec = load_model(checkpoint_path, graph_store, device)
    if int(run_spec["knowledge_cutoff"]) != cutoff:
        raise RuntimeError("checkpoint/config cutoff mismatch")
    if run_spec["provenance"]["config_sha256"] != sha256(config_path):
        raise RuntimeError("checkpoint/config hash mismatch")

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

    internal_ids, internal_cloud_scores = score_pointcloud_role(
        model, memories, point_config, "internal_test", provenance_mode,
        known_horizons, args.batch_size, args.points, args.workers,
        args.top_k, device,
    )
    internal_formula = add_frozen_ensemble(
        aggregate_by_key(internal_ids, internal_cloud_scores), frozen_cdf
    )
    internal_groups = formula_to_scientific_groups(internal_formula, formula_rows)
    internal_strata = {
        horizon: formula_to_scientific_groups(
            {
                key: value for key, value in internal_formula.items()
                if formula_rows[int(key)]["horizon"] == horizon
            },
            formula_rows,
        )
        for horizon in known_horizons
    }

    temporal_groups = None
    if cutoff == 1899:
        temporal_ids, temporal_cloud_scores = score_pointcloud_role(
            model, memories, point_config, "temporal_test", provenance_mode,
            ["pre1950"], args.batch_size, args.points, args.workers,
            args.top_k, device,
        )
        temporal_formula = add_frozen_ensemble(
            aggregate_by_key(temporal_ids, temporal_cloud_scores), frozen_cdf
        )
        temporal_groups = formula_to_scientific_groups(
            temporal_formula, formula_rows
        )

    if cutoff == 1899:
        curated_paths = {
            "curated_1901_1950": Path(config["tests"]["curated_1901_1950"]),
            "curated_post1950": Path(config["tests"]["curated_post1950"]),
        }
    else:
        forward_path = Path(config["tests"]["future_post1950"])
        curated_paths = {
            # Scored as a promoted/known clock-check set, never as a Phase-B
            # positive or tuning set.
            "curated_1901_1950": forward_path.with_name("post1900_to_1950.npz"),
            "curated_post1950": forward_path,
        }
    curated = {}
    for name, path in curated_paths.items():
        family_ids, record_scores = score_curated_archive(
            model, memories, path, args.batch_size, args.points, args.top_k, device
        )
        curated[name] = add_frozen_ensemble(
            aggregate_by_key(family_ids, record_scores), frozen_cdf
        )
    curated_all = None
    if cutoff == 1899:
        if set(curated["curated_1901_1950"]) & set(curated["curated_post1950"]):
            raise RuntimeError("curated temporal family partitions overlap")
        curated_all = {
            **curated["curated_1901_1950"],
            **curated["curated_post1950"],
        }

    metric = "one_class_ensemble"
    negative = metric_values(internal_groups, metric)
    comparisons = {}
    if temporal_groups is not None:
        comparisons["pre1900_internal_vs_generated_1901_1950"] = (
            clustered_bootstrap_auc(
                negative, metric_values(temporal_groups, metric),
                args.bootstrap, args.seed,
            )
        )
    if cutoff == 1899:
        comparisons["known_internal_vs_curated_1901_1950"] = (
            clustered_bootstrap_auc(
                negative, metric_values(curated["curated_1901_1950"], metric),
                args.bootstrap, args.seed + 1,
            )
        )
        comparisons["known_internal_vs_curated_all_post1900"] = (
            clustered_bootstrap_auc(
                negative, metric_values(curated_all, metric),
                args.bootstrap, args.seed + 3,
            )
        )
    comparisons["known_internal_vs_curated_post1950"] = clustered_bootstrap_auc(
        negative, metric_values(curated["curated_post1950"], metric),
        args.bootstrap, args.seed + 2,
    )
    if cutoff == 1950:
        for offset, (horizon, rows) in enumerate(internal_strata.items(), 3):
            comparisons[
                f"known_{horizon}_internal_vs_curated_post1950"
            ] = clustered_bootstrap_auc(
                metric_values(rows, metric),
                metric_values(curated["curated_post1950"], metric),
                args.bootstrap, args.seed + offset,
            )

    component_pairs = {
        "known_internal_vs_curated_post1950": curated["curated_post1950"],
    }
    if temporal_groups is not None:
        component_pairs[
            "pre1900_internal_vs_generated_1901_1950"
        ] = temporal_groups
    if cutoff == 1899:
        component_pairs["known_internal_vs_curated_1901_1950"] = (
            curated["curated_1901_1950"]
        )
        component_pairs["known_internal_vs_curated_all_post1900"] = curated_all
    component_aurocs = {
        label: {
            score_name: auroc(
                metric_values(internal_groups, score_name),
                metric_values(positive_rows, score_name),
            )
            for score_name in next(iter(internal_groups.values()))
        }
        for label, positive_rows in component_pairs.items()
    }

    result = {
        "status": "PASS",
        "schema_version": "3.0",
        "run_id": config["run_id"],
        "knowledge_cutoff": cutoff,
        "primary_metric": metric,
        "future_horizon_used_for_training_or_score_fitting": False,
        "query_input": "normalized unordered point cloud only",
        "aggregation": "cloud median to formula; formula median to source-law family/relation group",
        "memory": {
            "known_graph_formula_embeddings": int(len(memories["graph_formula"])),
            "known_graph_identity_embeddings": int(len(memories["graph_identity"])),
            "known_point_formula_prototypes": int(len(memories["point_formula"])),
            "point_gaussian_shrinkage": float(
                memories["point_gaussian"]["shrinkage"]
            ),
            "all_corpus_identity_labels": len(identity_names),
            "top_k": args.top_k,
        },
        "provenance": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "formula_table_sha256": sha256(formula_table_path),
        },
        "calibration_formulas": len(calibration_formula),
        "sets": {
            "known_internal": compact_rows(internal_groups),
            **{
                f"known_internal_{horizon}": compact_rows(rows)
                for horizon, rows in internal_strata.items()
            },
            **(
                {"generated_1901_1950": compact_rows(temporal_groups)}
                if temporal_groups is not None else {}
            ),
            **{name: compact_rows(rows) for name, rows in curated.items()},
            **(
                {"curated_all_post1900": compact_rows(curated_all)}
                if curated_all is not None else {}
            ),
        },
        "comparisons": comparisons,
        "component_aurocs_descriptive": component_aurocs,
        "limitations": [
            "AUROC evaluates temporal formula-family separation, not causal attribution to new physics.",
            "The historical negative uncertainty is governed by the number of held-out source-law families, not by cloud replicates.",
            "The 1901-1950 generator corpus is time-gated but not a complete bibliographically dated-source audit.",
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "knowledge_cutoff": cutoff,
        "comparisons": comparisons,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
