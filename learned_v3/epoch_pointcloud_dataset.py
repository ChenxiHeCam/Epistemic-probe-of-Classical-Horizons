"""PyTorch loader for immutable EPOCH mmap shards and dual-horizon views."""

from __future__ import annotations

import argparse
import bisect
import functools
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


PROVENANCE_MODES = {
    "strict": {"audited", "temporal_candidate"},
    "anchored": {"audited", "registry_named", "temporal_candidate"},
    "expanded": {"audited", "registry_named", "source_unresolved", "temporal_candidate"},
}


class EpochPointCloudDataset(Dataset):
    def __init__(
        self,
        bundle: str | Path,
        view_index: str | Path,
        view: str,
        role: str,
        provenance_mode: str = "expanded",
        horizons: Sequence[str] | None = None,
        points_per_sample: int = 64,
        seed: int = 20260907,
    ) -> None:
        self.bundle = Path(bundle).resolve()
        self.view_root = Path(view_index).resolve()
        manifest = json.loads(
            (self.view_root / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "PASS":
            raise RuntimeError("view index is not frozen PASS")
        if provenance_mode not in PROVENANCE_MODES:
            raise ValueError(f"unknown provenance mode: {provenance_mode}")
        allowed_tiers = PROVENANCE_MODES[provenance_mode]
        allowed_horizons = set(horizons) if horizons else None
        self.entries = [
            row for row in manifest["entries"]
            if row["view"] == view
            and row["role"] == role
            and row["tier"] in allowed_tiers
            and (allowed_horizons is None or row["horizon"] in allowed_horizons)
        ]
        if not self.entries:
            raise ValueError(
                f"empty dataset for view={view}, role={role}, mode={provenance_mode}"
            )
        self.points_per_sample = int(points_per_sample)
        if self.points_per_sample <= 0:
            raise ValueError("points_per_sample must be positive")
        self.seed = int(seed)
        self.epoch = 0
        self.offsets = [0]
        for entry in self.entries:
            self.offsets.append(self.offsets[-1] + int(entry["selected_clouds"]))

    def __len__(self) -> int:
        return self.offsets[-1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @functools.lru_cache(maxsize=16)
    def _arrays(self, source_shard: str):
        root = self.bundle / source_shard
        return {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (
                "values", "affine", "formula_index", "varying_index",
                "seed", "noise", "sampling_code", "parameters",
            )
        }

    @functools.lru_cache(maxsize=32)
    def _selection(self, relative: str):
        return np.load(self.view_root / relative, mmap_mode="r", allow_pickle=False)

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_index = bisect.bisect_right(self.offsets, index) - 1
        return entry_index, index - self.offsets[entry_index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry_index, local = self._locate(index)
        entry = self.entries[entry_index]
        selection = entry["selection"]
        if selection["mode"] == "all":
            row = local
        else:
            row = int(self._selection(selection["file"])[local])
        arrays = self._arrays(entry["source_shard"])
        values = np.asarray(arrays["values"][row])
        count = len(values)
        cloud_seed = int(arrays["seed"][row])
        rng = np.random.default_rng(
            cloud_seed ^ self.seed ^ ((self.epoch + 1) * 0x9E3779B97F4A7C15)
        )
        if count >= self.points_per_sample:
            chosen = np.sort(rng.choice(count, self.points_per_sample, replace=False))
        else:
            chosen = rng.choice(count, self.points_per_sample, replace=True)
        points = np.array(values[chosen], dtype=np.float32, copy=True)
        return {
            "points": torch.from_numpy(points),
            "formula_index": torch.tensor(int(arrays["formula_index"][row]), dtype=torch.long),
            "varying_index": torch.tensor(int(arrays["varying_index"][row]), dtype=torch.long),
            "affine": torch.from_numpy(
                np.array(arrays["affine"][row], dtype=np.float32, copy=True)
            ),
            "parameters": torch.from_numpy(
                np.array(arrays["parameters"][row], dtype=np.float32, copy=True)
            ),
            "noise": torch.tensor(float(arrays["noise"][row]), dtype=torch.float32),
            "sampling_code": torch.tensor(int(arrays["sampling_code"][row]), dtype=torch.long),
            "cloud_seed": torch.tensor(cloud_seed & ((1 << 63) - 1), dtype=torch.long),
        }


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle within/between shards while keeping each batch mmap-local."""

    def __init__(
        self,
        dataset: EpochPointCloudDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 20260907,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                (self.dataset.offsets[i + 1] - self.dataset.offsets[i]) // self.batch_size
                for i in range(len(self.dataset.entries))
            )
        return sum(
            math.ceil(
                (self.dataset.offsets[i + 1] - self.dataset.offsets[i]) / self.batch_size
            )
            for i in range(len(self.dataset.entries))
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        entry_order = np.arange(len(self.dataset.entries))
        if self.shuffle:
            rng.shuffle(entry_order)
        batches = []
        for entry_index in entry_order:
            start, stop = (
                self.dataset.offsets[entry_index], self.dataset.offsets[entry_index + 1]
            )
            indices = np.arange(start, stop)
            if self.shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset: offset + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def make_loader(
    bundle: str | Path,
    view_index: str | Path,
    view: str,
    role: str,
    provenance_mode: str,
    batch_size: int,
    points_per_sample: int = 64,
    workers: int = 4,
    shuffle: bool = True,
    seed: int = 20260907,
    horizons: Sequence[str] | None = None,
) -> tuple[EpochPointCloudDataset, ShardBatchSampler, DataLoader]:
    dataset = EpochPointCloudDataset(
        bundle=bundle,
        view_index=view_index,
        view=view,
        role=role,
        provenance_mode=provenance_mode,
        horizons=horizons,
        points_per_sample=points_per_sample,
        seed=seed,
    )
    sampler = ShardBatchSampler(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=shuffle, seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return dataset, sampler, loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--view-index", type=Path, required=True)
    parser.add_argument("--view", choices=("pre1900_temporal_eval", "pre1950_joint"), required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--provenance-mode", choices=tuple(PROVENANCE_MODES), default="expanded")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--horizon", choices=("pre1900", "pre1950"))
    args = parser.parse_args()
    dataset, sampler, loader = make_loader(
        args.bundle, args.view_index, args.view, args.role,
        args.provenance_mode, args.batch_size, args.points, args.workers,
        shuffle=False, horizons=[args.horizon] if args.horizon else None,
    )
    batch = next(iter(loader))
    print(json.dumps({
        "status": "PASS",
        "records": len(dataset),
        "batches": len(sampler),
        "points_shape": list(batch["points"].shape),
        "formula_index_min": int(batch["formula_index"].min()),
        "formula_index_max": int(batch["formula_index"].max()),
        "finite": bool(torch.isfinite(batch["points"]).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
