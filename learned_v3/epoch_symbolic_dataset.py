"""Streaming loader for EPOCH symbolic pretraining layers."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


PROFILE_LAYERS = {
    "strict": {
        "pre1900_audited_direct",
        "pre1950_temporal_direct",
    },
    "component_anchored": {
        "pre1900_math_audited_anchor",
        "pre1900_math_unresolved_anchor",
        "pre1950_temporal_math_anchored",
    },
    "expanded": {
        "pre1900_math_audited_anchor",
        "pre1900_math_unresolved_anchor",
        "pre1900_math_surface_screened",
        "pre1950_temporal_math_anchored",
        "pre1950_temporal_math_unanchored",
    },
}
VIEW_FIELDS = {
    "pre1900_temporal_eval": "pre1900_temporal_eval_role",
    "pre1950_joint": "pre1950_joint_role",
}


class EpochSymbolicDataset(IterableDataset):
    def __init__(
        self,
        bundle: str | Path,
        view: str,
        role: str,
        profile: str = "expanded",
        direct_bundle: str | Path | None = None,
        chain_bundle: str | Path | None = None,
        shuffle_buffer: int = 8192,
        seed: int = 20260907,
    ) -> None:
        super().__init__()
        expanded_bundle = Path(bundle).resolve()
        if profile == "strict":
            if direct_bundle is None:
                raise ValueError("profile=strict requires --direct-bundle")
            self.bundle = Path(direct_bundle).resolve()
            manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "PASS":
                raise RuntimeError("direct symbolic bundle is not frozen PASS")
        else:
            self.bundle = expanded_bundle
            manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
            audit = json.loads((self.bundle / "audit_report.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "PASS" or audit.get("status") != "PASS":
                raise RuntimeError("symbolic bundle is not frozen PASS")
        if view not in VIEW_FIELDS:
            raise ValueError(f"unknown view: {view}")
        if profile not in PROFILE_LAYERS:
            raise ValueError(f"unknown profile: {profile}")
        self.view = view
        self.role = role
        self.sources = [(self.bundle, layer) for layer in sorted(PROFILE_LAYERS[profile])]
        if chain_bundle is not None:
            chain_root = Path(chain_bundle).resolve()
            chain_manifest = json.loads(
                (chain_root / "manifest.json").read_text(encoding="utf-8")
            )
            if chain_manifest.get("status") != "PASS":
                raise RuntimeError("symbolic chain bundle is not frozen PASS")
            chain_layers = {
                "strict": {
                    "pre1900_audited_chain_core", "pre1950_temporal_chain_core",
                },
                "component_anchored": {
                    "pre1900_audited_chain_core", "pre1950_temporal_chain_core",
                },
                "expanded": {
                    "pre1900_audited_chain_core",
                    "pre1900_unresolved_chain_expanded",
                    "pre1950_temporal_chain_core",
                },
            }[profile]
            self.sources.extend((chain_root, layer) for layer in sorted(chain_layers))
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _eligible_rows(self) -> Iterator[dict]:
        field = VIEW_FIELDS[self.view]
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        eligible_index = 0
        for source_root, layer in self.sources:
            path = source_root / f"{layer}.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row[field] != self.role:
                        continue
                    if eligible_index % worker_count == worker_id:
                        yield row
                    eligible_index += 1

    def __iter__(self) -> Iterator[dict]:
        rows = self._eligible_rows()
        if self.shuffle_buffer <= 1:
            yield from rows
            return
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + self.epoch * 1009 + worker_id)
        buffer = []
        for row in rows:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer


def collate_symbolic(rows: list[dict]) -> dict:
    return {
        "task": [row["task"] for row in rows],
        "inputs": [row["inputs"] for row in rows],
        "context": [row.get("context") or [] for row in rows],
        "target": [row["target"] for row in rows],
        "augmentation_group_id": [row["augmentation_group_id"] for row in rows],
        "corpus_layer": [row["corpus_layer"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--direct-bundle", type=Path)
    parser.add_argument("--chain-bundle", type=Path)
    parser.add_argument("--view", choices=tuple(VIEW_FIELDS), required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_LAYERS), default="expanded")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    dataset = EpochSymbolicDataset(
        args.bundle, args.view, args.role, args.profile, args.direct_bundle,
        args.chain_bundle,
        shuffle_buffer=128,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_symbolic,
    )
    batch = next(iter(loader))
    print(json.dumps({
        "status": "PASS",
        "batch": len(batch["task"]),
        "tasks": sorted(set(batch["task"])),
        "layers": sorted(set(batch["corpus_layer"])),
        "unique_groups": len(set(batch["augmentation_group_id"])),
        "example": {
            "inputs": batch["inputs"][0],
            "context": batch["context"][0],
            "target": batch["target"][0],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
