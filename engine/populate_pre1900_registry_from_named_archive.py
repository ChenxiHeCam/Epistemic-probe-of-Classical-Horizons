"""Populate Paper2's fail-closed registry from exact, cited law-name matches.

This is intentionally conservative.  A record is admitted only when it is
already present in the archived encoder corpus and its normalized master name
exactly matches an alias in PAPER3's historical source registry.  Keyword-clean
records with no named-source match remain excluded.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MASTER = WORKSPACE / "dataset_20260531" / "_extract_master" / "master_20260616" / "master_nodes.jsonl"
ARCHIVE = ROOT / "data" / "classical_pointclouds.npz"
SOURCE_REGISTRY = WORKSPACE / "PAPER3_ABDUCTION" / "pre1900_source_registry_v1.json"
VERIFIED_CHAINS = WORKSPACE / "PAPER3_ABDUCTION" / "PRE1900_VERIFIED_CHAINS.jsonl"
OUTPUT = ROOT / "data" / "pre1900_registry.json"
REPORT = ROOT / "results" / "pre1900_registry_population_report.json"

MODERN = re.compile(
    r"eddington|bondi|schwarzschild|einstein|friedmann|hubble|chandrasekhar|hawking|"
    r"kolmogorov|prandtl|nusselt|debye|fermi|bose|dirac|schrod|pauli|heisenberg|"
    r"quantum|relativ|photon|supercond|semiconduct|laser|nuclear|radioactiv|compton|"
    r"broglie|tunnel|neutrino|gauge|standard model|bcs|ginzburg|landau|mhd|magnetohydro",
    re.I,
)


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def normalized_expression(value: str) -> str:
    value = re.sub(r"\s+", "", str(value or "")).replace("^", "**")
    value = value.replace("+-", "-")
    if value.endswith("=0"):
        value = value[:-2]
    return value


def main() -> None:
    source_bytes = SOURCE_REGISTRY.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    aliases: dict[str, dict] = {}
    for law in source["laws"]:
        if int(law["first_valid_year"]) > int(source["cutoff_year"]):
            raise ValueError(f"post-cutoff source entry: {law['family_id']}")
        for name in law["aliases"]:
            key = normalized_name(name)
            old = aliases.get(key)
            if old and old["family_id"] != law["family_id"]:
                raise ValueError(f"ambiguous normalized alias: {name}")
            aliases[key] = law

    expression_families: dict[str, set[str]] = collections.defaultdict(set)
    for raw in VERIFIED_CHAINS.read_text(encoding="utf-8").splitlines():
        chain = json.loads(raw)
        law = aliases.get(normalized_name(chain.get("root_name", "")))
        if not law or MODERN.search(json.dumps(chain, ensure_ascii=False)):
            continue
        values = [*(chain.get("root_exprs") or []), chain.get("law_expr", "")]
        for step in chain.get("steps") or []:
            values.extend(step.get("premises") or [])
            values.append(step.get("result", ""))
        for value in values:
            key = normalized_expression(value)
            if key:
                expression_families[key].add(law["family_id"])

    archive = np.load(ARCHIVE, allow_pickle=True)
    wanted = {str(value) for value in archive["ids"]}
    matched: dict[str, list[str]] = collections.defaultdict(list)
    match_route: dict[str, str] = {}
    rejected_modern = []
    master_hash = hashlib.sha256()
    archive_records: dict[str, dict] = {}
    with MASTER.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            master_hash.update(raw)
            try:
                record = json.loads(raw)
            except Exception:
                continue
            identifier = str(record.get("id", ""))
            if identifier not in wanted:
                continue
            archive_records[identifier] = {
                "id": identifier,
                "name": record.get("name", ""),
                "expr": record.get("expr", ""),
                "ancestor_seed_ids": [str(v) for v in record.get("ancestor_seed_ids") or []],
            }

    # Derived archive records often have blank names.  Trace them to named source
    # seeds, but admit only when every listed seed resolves to one cited family.
    ancestor_ids = {
        seed for record in archive_records.values()
        for seed in record["ancestor_seed_ids"]
    }
    seed_laws: dict[str, dict] = {}
    with MASTER.open("rb") as handle:
        for raw in handle:
            try:
                record = json.loads(raw)
            except Exception:
                continue
            identifier = str(record.get("id", ""))
            if identifier not in ancestor_ids:
                continue
            law = aliases.get(normalized_name(record.get("name", "")))
            if law and not MODERN.search(f"{record.get('name', '')} {record.get('expr', '')}"):
                seed_laws[identifier] = law

    unresolved_ancestor = 0
    mixed_family_ancestor = 0
    for identifier, record in archive_records.items():
        combined = f"{record['name']} {record['expr']}"
        if MODERN.search(combined):
            continue
        law = aliases.get(normalized_name(record["name"]))
        route = "direct_exact_name"
        if not law:
            families = expression_families.get(normalized_expression(record["expr"]), set())
            if len(families) == 1:
                law = next(item for item in source["laws"] if item["family_id"] == next(iter(families)))
                route = "exact_verified_chain_expression"
        if not law:
            seeds = record["ancestor_seed_ids"]
            if not seeds or any(seed not in seed_laws for seed in seeds):
                unresolved_ancestor += 1
                continue
            families = {seed_laws[seed]["family_id"] for seed in seeds}
            if len(families) != 1:
                mixed_family_ancestor += 1
                continue
            law = seed_laws[seeds[0]]
            route = "all_ancestor_seeds_exact_named_same_family"
        matched[law["family_id"]].append(identifier)
        match_route[identifier] = route

    laws = []
    source_by_family = {law["family_id"]: law for law in source["laws"]}
    for family_id in sorted(matched):
        law = source_by_family[family_id]
        laws.append({
            "family_id": family_id,
            "law_name": law["law_name"],
            "first_valid_year": int(law["first_valid_year"]),
            "citation": law["citation"],
            "record_ids": sorted(set(matched[family_id])),
            "audit_notes": (
                "Admitted by exact normalized master-name match, or by complete ancestry to exact-named "
                "seeds from one family, using PAPER3_ABDUCTION/pre1900_source_registry_v1.json; "
                "no keyword or domain heuristic fallback."
            ),
        })
    registry = {
        "status": "populated fail-closed pre-1900 registry; exact named-law matches only",
        "cutoff_year": int(source["cutoff_year"]),
        "policy": (
            "Every admitted record has an explicit ID, source-law family, first-valid year and citation. "
            "Unnamed and merely keyword-clean records are excluded."
        ),
        "laws": laws,
    }
    if not laws:
        raise SystemExit("No exact historical matches found; refusing to replace the registry.")
    OUTPUT.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "archive_records": len(wanted),
        "archive_ids_seen_in_master": len(archive_records),
        "admitted_records": sum(len(law["record_ids"]) for law in laws),
        "admitted_families": len(laws),
        "records_by_family": {law["family_id"]: len(law["record_ids"]) for law in laws},
        "match_routes": dict(collections.Counter(match_route.values())),
        "ancestor_seed_ids": len(ancestor_ids),
        "ancestor_seed_ids_with_exact_historical_match": len(seed_laws),
        "verified_chain_expression_keys": len(expression_families),
        "records_with_unresolved_ancestry": unresolved_ancestor,
        "records_with_mixed_family_ancestry": mixed_family_ancestor,
        "rejected_modern_after_name_match": rejected_modern,
        "source_registry_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "master_sha256": master_hash.hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "limitation": "Exact name matching is high precision but low recall; excluded records are not evidence of post-1900 origin.",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
