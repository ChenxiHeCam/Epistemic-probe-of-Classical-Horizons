"""Build a source-audited pre-1900 formula snapshot from an explicit registry.

Unlike the legacy domain/keyword filter, this builder is fail-closed: absence of
an audited registry entry means exclusion.  The registry, not a name heuristic,
supplies the historical date and citation.  It also assigns a source-law family
used for grouped train/validation/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT.parent / "dataset_20260531" / "_extract_master" / "master_20260616" / "master_nodes.jsonl"


def validate_registry(registry: dict) -> dict[str, dict]:
    cutoff = int(registry.get("cutoff_year", 1899))
    indexed: dict[str, dict] = {}
    families: set[str] = set()
    for law in registry.get("laws", []):
        required = ("family_id", "law_name", "first_valid_year", "citation", "record_ids")
        missing = [key for key in required if not law.get(key)]
        if missing:
            raise ValueError(f"registry law missing {missing}: {law!r}")
        if int(law["first_valid_year"]) > cutoff:
            raise ValueError(f"post-cutoff law cannot enter registry: {law['law_name']}")
        family = str(law["family_id"])
        if family in families:
            raise ValueError(f"duplicate family_id: {family}")
        families.add(family)
        for record_id in law["record_ids"]:
            record_id = str(record_id)
            if record_id in indexed:
                raise ValueError(f"record assigned to multiple source laws: {record_id}")
            indexed[record_id] = law
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "pre1900_registry.json")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "pre1900_audited_records.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "pre1900_corpus_manifest.json")
    args = parser.parse_args()

    registry_bytes = args.registry.read_bytes()
    registry = json.loads(registry_bytes.decode("utf-8"))
    wanted = validate_registry(registry)
    if not wanted:
        raise SystemExit(
            "Registry admits zero records. Populate data/pre1900_registry.json with cited, dated "
            "source-law families and explicit master IDs; heuristic fallback is intentionally disabled."
        )

    found: dict[str, dict] = {}
    master_hash = hashlib.sha256()
    with args.master.open("rb") as handle:
        for raw_line in handle:
            master_hash.update(raw_line)
            try:
                record = json.loads(raw_line)
            except Exception:
                continue
            record_id = str(record.get("id", ""))
            if record_id in wanted:
                found[record_id] = record

    missing = sorted(set(wanted) - set(found))
    if missing:
        raise RuntimeError(f"{len(missing)} registered IDs are absent from the pinned master; first: {missing[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record_id in sorted(found):
            law = wanted[record_id]
            snapshot = {
                "record": found[record_id],
                "period_provenance": {
                    "family_id": law["family_id"],
                    "law_name": law["law_name"],
                    "first_valid_year": int(law["first_valid_year"]),
                    "citation": law["citation"],
                    "citation_url_or_doi": law.get("citation_url_or_doi"),
                    "audit_notes": law.get("audit_notes"),
                },
            }
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    family_counts: dict[str, int] = {}
    for law in wanted.values():
        family = str(law["family_id"])
        family_counts[family] = family_counts.get(family, 0) + 1
    manifest = {
        "status": "source-audited pre-1900 selection snapshot; point-cloud generation and grouped split not yet run",
        "cutoff_year": int(registry["cutoff_year"]),
        "records": len(found),
        "source_law_families": len(family_counts),
        "records_by_family": dict(sorted(family_counts.items())),
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "master_sha256": master_hash.hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "split_policy": "group exclusively by family_id; no family may cross train/validation/test",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
