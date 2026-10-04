"""Mmap-backed formula-graph store and dual-horizon PyTorch dataset for EPOCH."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


PROVENANCE_MODES = {
    "strict": {"audited", "temporal_candidate"},
    "anchored": {"audited", "registry_named", "temporal_candidate"},
    "expanded": {"audited", "registry_named", "source_unresolved", "temporal_candidate"},
}


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class EpochFormulaGraphStore:
    """Random-access immutable formula graphs shared by point-cloud samples."""

    ARRAY_NAMES = (
        "graph_ptr", "edge_ptr", "node_type", "node_constant",
        "node_variable_slot", "node_depth", "edge_source", "edge_target",
        "edge_direction", "edge_child_position",
    )

    def __init__(self, bundle: str | Path) -> None:
        self.bundle = Path(bundle).resolve()
        self.manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("status") != "PASS":
            raise RuntimeError("formula graph bundle is not frozen PASS")
        self.arrays = {
            name: np.load(
                self.bundle / self.manifest["files"][name]["file"],
                mmap_mode="r", allow_pickle=False,
            )
            for name in self.ARRAY_NAMES
        }
        self.metadata = _read_jsonl_gz(
            self.bundle / self.manifest["metadata"]["file"]
        )
        if len(self.metadata) != self.manifest["formula_count"]:
            raise RuntimeError("formula metadata cardinality mismatch")

    def __len__(self) -> int:
        return len(self.metadata)

    def graph(self, formula_index: int) -> dict[str, torch.Tensor]:
        formula_index = int(formula_index)
        if formula_index < 0 or formula_index >= len(self):
            raise IndexError(formula_index)
        ns = int(self.arrays["graph_ptr"][formula_index])
        ne = int(self.arrays["graph_ptr"][formula_index + 1])
        es = int(self.arrays["edge_ptr"][formula_index])
        ee = int(self.arrays["edge_ptr"][formula_index + 1])
        return {
            "node_type": torch.from_numpy(
                np.array(self.arrays["node_type"][ns:ne], dtype=np.int64, copy=True)
            ),
            "node_constant": torch.from_numpy(
                np.array(self.arrays["node_constant"][ns:ne], dtype=np.float32, copy=True)
            ),
            "node_variable_slot": torch.from_numpy(
                np.array(self.arrays["node_variable_slot"][ns:ne], dtype=np.int64, copy=True)
            ),
            "node_depth": torch.from_numpy(
                np.array(self.arrays["node_depth"][ns:ne], dtype=np.int64, copy=True)
            ),
            "edge_index": torch.from_numpy(np.stack((
                np.array(self.arrays["edge_source"][es:ee], dtype=np.int64, copy=True) - ns,
                np.array(self.arrays["edge_target"][es:ee], dtype=np.int64, copy=True) - ns,
            ))),
            "edge_direction": torch.from_numpy(
                np.array(self.arrays["edge_direction"][es:ee], dtype=np.int64, copy=True)
            ),
            "edge_child_position": torch.from_numpy(
                np.array(self.arrays["edge_child_position"][es:ee], dtype=np.int64, copy=True)
            ),
            "formula_index": torch.tensor(formula_index, dtype=torch.long),
        }

    def batch(
        self,
        formula_indices: Iterable[int] | torch.Tensor,
        deduplicate: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Collate graphs; optionally encode each repeated formula only once."""
        if isinstance(formula_indices, torch.Tensor):
            requested = [int(value) for value in formula_indices.detach().cpu().tolist()]
        else:
            requested = [int(value) for value in formula_indices]
        if not requested:
            raise ValueError("cannot collate an empty formula-index sequence")
        if deduplicate:
            graph_indices: list[int] = []
            lookup: dict[int, int] = {}
            sample_to_graph = []
            for formula_index in requested:
                if formula_index not in lookup:
                    lookup[formula_index] = len(graph_indices)
                    graph_indices.append(formula_index)
                sample_to_graph.append(lookup[formula_index])
        else:
            graph_indices = requested
            sample_to_graph = list(range(len(requested)))

        graphs = [self.graph(index) for index in graph_indices]
        node_offsets = [0]
        for graph in graphs:
            node_offsets.append(node_offsets[-1] + int(graph["node_type"].numel()))
        edge_parts = [
            graph["edge_index"] + node_offsets[i]
            for i, graph in enumerate(graphs)
        ]
        node_batch = [
            torch.full((graph["node_type"].numel(),), i, dtype=torch.long)
            for i, graph in enumerate(graphs)
        ]
        return {
            "node_type": torch.cat([graph["node_type"] for graph in graphs]),
            "node_constant": torch.cat([graph["node_constant"] for graph in graphs]),
            "node_variable_slot": torch.cat(
                [graph["node_variable_slot"] for graph in graphs]
            ),
            "node_depth": torch.cat([graph["node_depth"] for graph in graphs]),
            "edge_index": torch.cat(edge_parts, dim=1),
            "edge_direction": torch.cat(
                [graph["edge_direction"] for graph in graphs]
            ),
            "edge_child_position": torch.cat(
                [graph["edge_child_position"] for graph in graphs]
            ),
            "node_batch": torch.cat(node_batch),
            "graph_ptr": torch.tensor(node_offsets, dtype=torch.long),
            "graph_formula_index": torch.tensor(graph_indices, dtype=torch.long),
            "sample_to_graph": torch.tensor(sample_to_graph, dtype=torch.long),
        }


class EpochFormulaGraphDataset(Dataset):
    """Formula-level horizon view used for graph-only pretraining/evaluation."""

    def __init__(
        self,
        bundle: str | Path,
        view: str,
        role: str,
        provenance_mode: str = "expanded",
        horizons: Sequence[str] | None = None,
    ) -> None:
        if view not in {"pre1900_temporal_eval", "pre1950_joint"}:
            raise ValueError(f"unknown view: {view}")
        if provenance_mode not in PROVENANCE_MODES:
            raise ValueError(f"unknown provenance mode: {provenance_mode}")
        self.store = EpochFormulaGraphStore(bundle)
        role_key = f"{view}_role"
        tiers = PROVENANCE_MODES[provenance_mode]
        allowed_horizons = set(horizons) if horizons else None
        self.formula_indices = [
            int(row["formula_index"])
            for row in self.store.metadata
            if row[role_key] == role
            and row["tier"] in tiers
            and (allowed_horizons is None or row["horizon"] in allowed_horizons)
        ]
        if not self.formula_indices:
            raise ValueError(
                f"empty graph dataset for view={view}, role={role}, mode={provenance_mode}"
            )

    def __len__(self) -> int:
        return len(self.formula_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.store.graph(self.formula_indices[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--view", choices=("pre1900_temporal_eval", "pre1950_joint"), required=True
    )
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--provenance-mode", choices=tuple(PROVENANCE_MODES), default="expanded"
    )
    parser.add_argument("--horizon", choices=("pre1900", "pre1950"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    dataset = EpochFormulaGraphDataset(
        args.bundle, args.view, args.role, args.provenance_mode,
        [args.horizon] if args.horizon else None,
    )
    take = min(args.batch_size, len(dataset))
    indices = dataset.formula_indices[:take]
    batch = dataset.store.batch(indices)
    edge_index = batch["edge_index"]
    print(json.dumps({
        "status": "PASS",
        "formula_records": len(dataset),
        "batch_graphs": int(batch["graph_formula_index"].numel()),
        "batch_nodes": int(batch["node_type"].numel()),
        "batch_directed_edges": int(edge_index.shape[1]),
        "edge_index_in_bounds": bool(
            edge_index.numel() == 0
            or (int(edge_index.min()) >= 0 and int(edge_index.max()) < batch["node_type"].numel())
        ),
        "finite_constants": bool(torch.isfinite(batch["node_constant"]).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
