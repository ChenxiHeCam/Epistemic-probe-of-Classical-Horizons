"""Inventory downloaded NIST files without assigning physics-boundary labels.

The audit deliberately stops at structural facts (hash, numeric block size and
column count).  Scientific use still requires an independently specified
incumbent model, measurement model and mechanism-level grouping.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$")


def numeric_tokens(line: str) -> list[str] | None:
    tokens = line.replace(",", " ").split()
    if len(tokens) < 1 or not all(NUMBER.fullmatch(token) for token in tokens):
        return None
    return tokens


def largest_numeric_block(lines: list[str]) -> dict:
    """Return the largest consecutive, fixed-width numeric block."""
    blocks: list[tuple[int, int, int]] = []
    start = None
    width = None
    for idx, line in enumerate(lines + ["END"]):
        tokens = numeric_tokens(line)
        this_width = len(tokens) if tokens else None
        if this_width is not None and (start is None or this_width == width):
            if start is None:
                start, width = idx, this_width
            continue
        if start is not None:
            blocks.append((start, idx, int(width)))
            start, width = None, None
        if this_width is not None:
            start, width = idx, this_width
    if not blocks:
        return {"start_line": None, "end_line": None, "rows": 0, "columns": 0}
    start, end, width = max(blocks, key=lambda item: (item[1] - item[0], item[2]))
    return {"start_line": start + 1, "end_line": end, "rows": end - start, "columns": width}


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for entry in manifest["datasets"]:
        path = RAW / f"{entry['id']}.DAT"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        hash_to_ids[digest].append(entry["id"])
        text = payload.decode("latin-1")
        block = largest_numeric_block(text.splitlines())
        rows.append({
            **entry,
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": len(payload),
            "sha256": digest,
            **block,
            "structurally_parseable": block["rows"] >= 3 and block["columns"] >= 1,
            "abscissa_policy": "explicit columns require semantic audit" if block["columns"] >= 2
            else "single series; sample index is only a provisional abscissa",
            "physics_label_frozen": False,
        })

    role_counts = Counter(row["role"] for row in rows)
    group_counts = Counter(row["group"] for row in rows)
    duplicate_files = [ids for ids in hash_to_ids.values() if len(ids) > 1]
    report = {
        "status": "development inventory; not a prospectively sealed evaluation",
        "source_index": manifest["source_index"],
        "files_expected": len(manifest["datasets"]),
        "files_present": len(rows),
        "files_structurally_parseable": sum(row["structurally_parseable"] for row in rows),
        "exact_duplicate_file_groups": duplicate_files,
        "role_counts": dict(sorted(role_counts.items())),
        "mechanism_group_counts": dict(sorted(group_counts.items())),
        "audit_boundary": (
            "Numeric blocks are inventoried only. No anomaly/control label, model, p-value, "
            "or mechanism-independence claim is assigned without a frozen horizon and measurement model."
        ),
        "datasets": rows,
    }
    out = ROOT / "inventory_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "files_expected", "files_present", "files_structurally_parseable",
        "exact_duplicate_file_groups", "role_counts")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
