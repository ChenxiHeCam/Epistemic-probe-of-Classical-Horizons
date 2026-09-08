"""Portable inference for the frozen 1899-horizon EPOCH point encoder.

The scorer accepts only unordered numeric (x, y) point clouds.  Formula
graphs are used during pretraining, but are not required by this query path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from epoch_learned_v3 import EpochArchitecture, PointCloudEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "epoch_1899_point_student.pt"
DEFAULT_MEMORY = ROOT / "models" / "epoch_1899_inference_memory.pt"


def robust_normalize(clouds: np.ndarray) -> np.ndarray:
    """Apply the location/scale convention frozen for temporal evaluation."""
    clouds = np.asarray(clouds, dtype=np.float32)
    if clouds.ndim == 2:
        clouds = clouds[None, ...]
    if clouds.ndim != 3 or clouds.shape[-1] != 2:
        raise ValueError("expected an array with shape (records, points, 2)")
    location = np.median(clouds, axis=1, keepdims=True)
    q25 = np.quantile(clouds, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(clouds, 0.75, axis=1, keepdims=True)
    scale = q75 - q25
    fallback = np.std(clouds, axis=1, keepdims=True)
    threshold = 1e-9 * (np.abs(location) + 1.0)
    scale = np.where(np.isfinite(scale) & (scale > threshold), scale, fallback)
    normalized = (clouds - location) / np.maximum(scale, 1e-8)
    if not np.isfinite(normalized).all():
        raise ValueError("point cloud contains non-finite or zero-scale values")
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


class Epoch1899Scorer:
    """Frozen positive-only one-class scorer for unordered point clouds."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        memory_path: Path = DEFAULT_MEMORY,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        architecture = EpochArchitecture(**bundle["architecture"])
        self.model = PointCloudEncoder(architecture)
        self.model.load_state_dict(bundle["state_dict"], strict=True)
        self.model.to(self.device).eval()

        frozen = torch.load(memory_path, map_location="cpu", weights_only=False)
        memories = frozen["memories"]
        self.memories = {
            name: memories[name].to(self.device).float()
            for name in ("graph_formula", "graph_identity", "point_formula")
        }
        self.memories["point_gaussian"] = {
            "mean": memories["point_gaussian"]["mean"].to(self.device).float(),
            "precision": memories["point_gaussian"]["precision"].to(self.device).float(),
        }
        self.cdf = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in frozen["cdf"].items()
        }
        self.points = int(frozen["identity"]["points"])
        self.top_k = int(frozen["identity"]["top_k"])
        self.identity = dict(frozen["identity"])

    @torch.inference_mode()
    def score(
        self,
        clouds: np.ndarray,
        record_ids: np.ndarray | None = None,
        batch_size: int = 64,
    ) -> list[dict[str, float]]:
        clouds = robust_normalize(clouds)
        if record_ids is None:
            record_ids = np.asarray([f"record-{i}" for i in range(len(clouds))])
        record_ids = np.asarray(record_ids).astype(str)
        if len(record_ids) != len(clouds):
            raise ValueError("record_ids and clouds have different lengths")
        clouds = deterministic_subsample(clouds, record_ids, self.points)

        rows: list[dict[str, float]] = []
        for start in range(0, len(clouds), batch_size):
            batch = torch.from_numpy(clouds[start : start + batch_size]).to(self.device)
            mask = torch.ones(batch.shape[:2], dtype=torch.bool, device=self.device)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with context:
                embedding = self.model(batch, mask)
            query = F.normalize(embedding.float(), dim=-1)
            raw: dict[str, np.ndarray] = {}
            for name in ("graph_formula", "graph_identity", "point_formula"):
                memory = self.memories[name]
                k = min(self.top_k, len(memory))
                nearest = (query @ memory.T).topk(
                    k, dim=1, largest=True, sorted=False
                ).values
                raw[f"{name}_knn_distance"] = (
                    1.0 - nearest.mean(dim=1)
                ).cpu().numpy()
            gaussian = self.memories["point_gaussian"]
            delta = query - gaussian["mean"]
            raw["point_shrinkage_mahalanobis"] = torch.einsum(
                "bi,ij,bj->b", delta, gaussian["precision"], delta
            ).cpu().numpy()

            for offset in range(len(batch)):
                row = {name: float(values[offset]) for name, values in raw.items()}
                calibrated = []
                for name, reference in self.cdf.items():
                    percentile = float(
                        (np.searchsorted(reference, row[name], side="right") + 1.0)
                        / (len(reference) + 1.0)
                    )
                    row[f"calibrated_{name}"] = percentile
                    calibrated.append(percentile)
                row["one_class_ensemble"] = float(np.mean(calibrated))
                rows.append(row)
        return rows


def load_input(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        clouds = archive["X"]
        record_ids = (
            archive["record_ids"].astype(str)
            if "record_ids" in archive.files
            else np.asarray([f"record-{i}" for i in range(len(clouds))])
        )
        family_ids = (
            archive["family_ids"].astype(str)
            if "family_ids" in archive.files
            else None
        )
        return clouds, record_ids, family_ids

    delimiter = "," if path.suffix.lower() == ".csv" else None
    array = np.genfromtxt(path, delimiter=delimiter, comments="#")
    if array.ndim == 1:
        array = array[None, :]
    array = array[np.isfinite(array).all(axis=1)]
    if array.shape[1] < 2:
        raise ValueError("text input must contain at least two numeric columns")
    return array[:, :2][None, ...], np.asarray([path.stem]), None


def aggregate_families(
    rows: list[dict[str, float]],
    family_ids: np.ndarray,
    cdf: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(family_ids.astype(str)):
        grouped[family].append(index)
    result: dict[str, dict[str, float]] = {}
    for family, indices in grouped.items():
        row = {
            name: float(np.median([rows[index][name] for index in indices]))
            for name in cdf
        }
        calibrated = []
        for name, reference in cdf.items():
            percentile = float(
                (np.searchsorted(reference, row[name], side="right") + 1.0)
                / (len(reference) + 1.0)
            )
            row[f"calibrated_{name}"] = percentile
            calibrated.append(percentile)
        row["one_class_ensemble"] = float(np.mean(calibrated))
        result[family] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    clouds, record_ids, family_ids = load_input(args.input)
    scorer = Epoch1899Scorer(args.model, args.memory, args.device)
    rows = scorer.score(clouds, record_ids, args.batch_size)
    payload = {
        "schema_version": "3.0-portable-inference",
        "knowledge_cutoff": 1899,
        "query_input": "normalized unordered x-y point cloud only",
        "record_scores": [
            {"record_id": str(record_id), **row}
            for record_id, row in zip(record_ids, rows)
        ],
    }
    if family_ids is not None:
        payload["family_scores"] = aggregate_families(rows, family_ids, scorer.cdf)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
