"""Audit provenance and obvious post-1900 leakage in the encoder corpus."""

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
DATA = ROOT / "data" / "classical_pointclouds.npz"
OUTPUT = ROOT / "results" / "classical_corpus_provenance_audit.json"

# These terms are a high-precision screen, not a complete dating system.
OBVIOUS_POST1900 = re.compile(
    r"eddington|bondi|schwarzschild|einstein|friedmann|hubble|chandrasekhar|hawking|"
    r"kolmogorov|prandtl|nusselt|debye|fermi|bose|dirac|schrod|pauli|heisenberg|"
    r"quantum|relativ|photon|supercond|semiconduct|laser|nuclear|radioactiv|compton|"
    r"broglie|tunnel|neutrino|gauge|standard model|bcs|ginzburg|landau|mhd|magnetohydro",
    re.I,
)

YEAR_KEYS = {"year", "date", "publication_year", "first_year", "introduced", "era"}
SOURCE_KEYS = {"source", "citation", "doi", "reference", "references", "provenance", "url"}


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def normalize_expression(value):
    expression = (value or "").split("=", 1)[-1].lower()
    expression = re.sub(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", "#", expression)
    expression = re.sub(r"\s+", "", expression)
    return expression


def structural_template(expression, variables):
    value = normalize_expression(expression)
    symbols = sorted(
        [item.get("sym", "") for item in variables if isinstance(item, dict) and item.get("sym")],
        key=len, reverse=True,
    )
    for index, symbol in enumerate(symbols):
        value = re.sub(rf"\b{re.escape(symbol.lower())}\b", f"v{index}", value)
    return value


def main():
    archive = np.load(DATA, allow_pickle=True)
    ids = [str(value) for value in archive["ids"]]
    wanted = set(ids)
    records = {}
    master_hash = hashlib.sha256()
    with MASTER.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            master_hash.update(raw_line)
            try:
                record = json.loads(raw_line)
            except Exception:
                continue
            identifier = str(record.get("id", ""))
            if identifier in wanted:
                records[identifier] = record
            if line_number % 500_000 == 0:
                print(f"scanned {line_number:,}; matched {len(records):,}/{len(wanted):,}", flush=True)

    modern = []
    missing_year = 0
    missing_source = 0
    domains = collections.Counter()
    origins = collections.Counter()
    names = collections.Counter()
    expressions = collections.Counter()
    templates = collections.Counter()
    for identifier in ids:
        record = records.get(identifier)
        if record is None:
            continue
        name = str(record.get("name", "") or "")
        expression = str(record.get("expr", "") or "")
        combined = f"{name} {expression}"
        match = OBVIOUS_POST1900.search(combined)
        if match:
            modern.append({
                "id": identifier, "term": match.group(0), "name": name,
                "expr": expression, "domain": record.get("domain"), "origin": record.get("origin"),
            })
        if not any(record.get(key) not in (None, "", []) for key in YEAR_KEYS):
            missing_year += 1
        if not any(record.get(key) not in (None, "", []) for key in SOURCE_KEYS):
            missing_source += 1
        domains[str(record.get("domain", ""))] += 1
        origins[str(record.get("origin", ""))] += 1
        if normalize_name(name):
            names[normalize_name(name)] += 1
        expressions[normalize_expression(expression)] += 1
        templates[structural_template(expression, record.get("variables") or [])] += 1

    def duplicate_summary(counter):
        duplicate_groups = [count for key, count in counter.items() if key and count > 1]
        return {
            "unique_groups": sum(1 for key in counter if key),
            "duplicate_groups": len(duplicate_groups),
            "records_in_duplicate_groups": int(sum(duplicate_groups)),
            "largest_group": int(max(duplicate_groups, default=1)),
        }

    result = {
        "corpus_records": len(ids),
        "unique_archive_ids": len(wanted),
        "matched_to_master": len(records),
        "missing_master_records": len(wanted) - len(records),
        "records_without_year_metadata": missing_year,
        "records_without_source_metadata": missing_source,
        "obvious_post1900_keyword_hits": len(modern),
        "obvious_post1900_fraction": len(modern) / max(len(records), 1),
        "modern_examples": modern[:100],
        "exact_normalized_name_duplicates": duplicate_summary(names),
        "exact_normalized_expression_duplicates": duplicate_summary(expressions),
        "structural_template_duplicates": duplicate_summary(templates),
        "domains": domains.most_common(),
        "origins": origins.most_common(),
        "audit_limit": (
            "Keyword hits are a lower bound. Absence of a hit does not establish a pre-1900 date. "
            "The corpus cannot be certified period-pure without source-level dates and citations."
        ),
        "master_sha256": master_hash.hexdigest(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "modern_examples"}, indent=2))


if __name__ == "__main__":
    main()
