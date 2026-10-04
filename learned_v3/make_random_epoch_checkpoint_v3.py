"""Create an untrained, architecture-matched checkpoint for EPOCH v3.

The resulting checkpoint is intentionally compatible with the frozen evaluator
so the trained representation can be compared with deterministic random
features without changing any scoring or aggregation code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from epoch_formula_graph_dataset import EpochFormulaGraphStore
from epoch_learned_v3 import EpochArchitecture, EpochDualEncoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph_store = EpochFormulaGraphStore(config["formula_graph"]["bundle"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    architecture = EpochArchitecture()
    model = EpochDualEncoder(
        graph_store.manifest["node_vocab_size"], architecture
    ).cpu().eval()
    run_spec = {
        "schema_version": "3.0-random-baseline",
        "run_id": f"{config['run_id']}_untrained_random",
        "knowledge_cutoff": int(config["knowledge_cutoff"]),
        "architecture": architecture.to_dict(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimization": {
            "trained": False,
            "optimizer_steps": 0,
            "seed": args.seed,
        },
        "provenance": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "purpose": "untrained architecture-matched descriptive control",
        },
    }
    payload = {
        "schema_version": "3.0-random-baseline",
        "run_spec": run_spec,
        "model": model.state_dict(),
        "epoch": 0,
        "batch_in_epoch": 0,
        "global_batch": 0,
        "optimizer_step": 0,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "output": str(output),
        "sha256": sha256(output),
        "parameters": run_spec["parameters"],
        "knowledge_cutoff": run_spec["knowledge_cutoff"],
        "trained": False,
    }, indent=2))


if __name__ == "__main__":
    main()
