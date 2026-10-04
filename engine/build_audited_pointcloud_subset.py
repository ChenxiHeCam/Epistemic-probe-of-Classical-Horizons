"""Materialize the source-audited pre-1900 subset of the legacy point clouds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "pre1900_registry.json"
SOURCE_REGISTRY = ROOT.parent / "PAPER3_ABDUCTION" / "pre1900_source_registry_v1.json"
SOURCE = ROOT / "data" / "classical_pointclouds.npz"
OUTPUT = ROOT / "data" / "pre1900_audited_pointclouds.npz"
MANIFEST = ROOT / "results" / "pre1900_audited_pointclouds_manifest.json"


def build_family_splits() -> dict[str, str]:
    source = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    families = sorted(
        {str(law["family_id"]) for law in source["laws"]},
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    heldout = max(1, round(len(families) * 0.1))
    return {
        family: ("test" if index < heldout else "validation" if index < 2 * heldout else "train")
        for index, family in enumerate(families)
    }


def main() -> None:
    registry_bytes = REGISTRY.read_bytes()
    registry = json.loads(registry_bytes.decode("utf-8"))
    family_splits = build_family_splits()
    id_to_family = {}
    for law in registry["laws"]:
        for identifier in law["record_ids"]:
            if identifier in id_to_family:
                raise ValueError(f"record appears in multiple families: {identifier}")
            id_to_family[str(identifier)] = str(law["family_id"])

    source = np.load(SOURCE, allow_pickle=True)
    source_ids = [str(value) for value in source["ids"]]
    positions = [index for index, identifier in enumerate(source_ids) if identifier in id_to_family]
    selected_ids = [source_ids[index] for index in positions]
    missing = sorted(set(id_to_family) - set(selected_ids))
    if missing:
        raise RuntimeError(f"registered IDs absent from point-cloud archive: {missing[:5]}")
    families = np.asarray([id_to_family[value] for value in selected_ids])
    splits = np.asarray([family_splits[value] for value in families])
    payload = {
        key: source[key][positions]
        for key in ("X", "ids", "sigs", "dimsxy")
    }
    payload["family_ids"] = families
    payload["splits"] = splits
    np.savez_compressed(OUTPUT, **payload)

    split_families = {}
    for family, split in zip(families, splits):
        old = split_families.setdefault(str(family), str(split))
        if old != split:
            raise AssertionError(f"family crosses splits: {family}")
    manifest = {
        "status": "source-audited pre-1900 point-cloud subset; no post-1900 or heuristic-only rows admitted",
        "records": len(selected_ids),
        "families": len(set(families)),
        "records_by_split": {name: int(np.sum(splits == name)) for name in ("train", "validation", "test")},
        "families_by_split": {
            name: len({str(family) for family, split in zip(families, splits) if split == name})
            for name in ("train", "validation", "test")
        },
        "family_split_policy": "shared deterministic 80/10/10 mapping over all families in pre1900_source_registry_v1.json",
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "source_archive_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "shape": list(payload["X"].shape),
        "training_warning": (
            "This high-precision subset is suitable for leakage-free experiments but is too small "
            "to support a strong foundation-model claim without adding more source-audited families."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
