"""Fail-closed structural and leakage audit for the post-1900 benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BUILDER = ROOT / "engine" / "build_post1900_formula_benchmark.py"
REGISTRY = DATA / "post1900_formula_registry.json"
ARCHIVE = DATA / "post1900_formula_pointclouds.npz"
RECORDS = DATA / "post1900_formula_records.jsonl"
MANIFEST = RESULTS / "post1900_formula_benchmark_manifest.json"
OUTPUT = RESULTS / "post1900_formula_benchmark_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    generated = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = generated["laws"]
    source = runpy.run_path(str(BUILDER), run_name="post1900_builder_audit")
    source_rows = source["REGISTRY"]
    assert rows == source_rows, "generated registry drifted from executable source registry"

    family_ids = [row["family_id"] for row in rows]
    assert len(family_ids) == len(set(family_ids))
    assert all(isinstance(row["first_valid_year"], int) and row["first_valid_year"] > 1900 for row in rows)
    assert all(row["citation"].strip() and row["law_name"].strip() and row["equation"].strip() for row in rows)
    assert all(row["sampling"] in {"linear", "log"} for row in rows)
    assert all(len(row["x_range"]) == 2 and 0 < row["x_range"][1] - row["x_range"][0] for row in rows)

    citation_year_mismatches = []
    doi_families = []
    for row in rows:
        years = [int(year) for year in re.findall(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)", row["citation"])]
        if row["first_valid_year"] not in years:
            citation_year_mismatches.append({
                "family_id": row["family_id"], "declared": row["first_valid_year"],
                "citation_years": years,
            })
        if re.search(r"doi:10\.\d{4,9}/\S+", row["citation"], re.I):
            doi_families.append(row["family_id"])
    assert not citation_year_mismatches, citation_year_mismatches

    archive = np.load(ARCHIVE, allow_pickle=True)
    clouds = np.asarray(archive["X"])
    archive_families = np.asarray(archive["family_ids"]).astype(str)
    record_ids = np.asarray(archive["record_ids"]).astype(str)
    assert clouds.shape == (manifest["records"], manifest["points_per_record"], 2)
    assert len(record_ids) == len(set(record_ids))
    assert set(archive_families) == set(family_ids)
    counts = Counter(archive_families)
    assert set(counts.values()) == {manifest["records_per_family"]}
    assert np.all(np.isfinite(clouds))
    assert all(np.all(np.diff(cloud[:, 0]) >= 0) for cloud in clouds)
    assert all(len(np.unique(cloud[:, 0])) >= 0.95*cloud.shape[0] for cloud in clouds)
    assert all(np.std(cloud[:, 1]) > 1e-9 for cloud in clouds)

    metadata = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(metadata) == len(clouds)
    assert [row["record_id"] for row in metadata] == record_ids.tolist()
    assert len({row["seed"] for row in metadata}) == len(metadata)
    assert all(row["n"] == manifest["points_per_record"] for row in metadata)
    assert all(0.005 <= row["noise_fraction"] <= 0.025 for row in metadata)

    for name, info in manifest["files"].items():
        path = DATA / name
        assert path.stat().st_size == info["bytes"]
        assert sha256(path) == info["sha256"]

    pre = np.load(DATA / "pre1900_audited_pointclouds.npz", allow_pickle=True)
    pre_families = set(np.asarray(pre["family_ids"]).astype(str))
    identifier_overlap = sorted(pre_families.intersection(family_ids))
    assert not identifier_overlap

    signature_groups = {}
    for row in rows:
        signature_groups.setdefault(row["signature"], []).append(row["family_id"])
    duplicated_signatures = {
        signature: families for signature, families in signature_groups.items() if len(families) > 1
    }
    dictionary_overlap = [
        row["family_id"] for row in rows if row["shape_class"] == "declared_dictionary_overlap"
    ]
    assert len(dictionary_overlap) >= 10, "benchmark must retain later laws with old/simple shapes"

    result = {
        "status": "PASS",
        "scope": "offline structural, citation-field and leakage audit; not line-by-line historical source verification",
        "families": len(rows), "records": len(clouds),
        "strict_post1900": True,
        "year_range": [min(row["first_valid_year"] for row in rows), max(row["first_valid_year"] for row in rows)],
        "primary_citation_present": len(rows),
        "citation_year_matches_declared_year": len(rows),
        "citations_with_doi": len(doi_families),
        "citations_without_doi": len(rows)-len(doi_families),
        "family_identifier_overlap_with_pre1900_archive": identifier_overlap,
        "declared_dictionary_overlap_hard_cases": dictionary_overlap,
        "duplicated_functional_signatures": duplicated_signatures,
        "special_reduction_notes": generated["reduction_notes"],
        "numeric_checks": {
            "all_finite": True, "monotone_x": True,
            "at_least_95pct_unique_x_per_record": True, "nonconstant_y": True,
        },
        "hashes": {
            "builder": sha256(BUILDER), "registry": sha256(REGISTRY),
            "archive": sha256(ARCHIVE), "records": sha256(RECORDS),
        },
        "claim_boundary": generated["claim_boundary"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
