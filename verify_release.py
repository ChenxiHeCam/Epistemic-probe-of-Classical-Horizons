"""Verify byte identities for the portable EPOCH repository release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "release_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for relative, identity in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != identity["bytes"]:
            failures.append(f"size: {relative}")
            continue
        if sha256(path) != identity["sha256"]:
            failures.append(f"sha256: {relative}")
    if failures:
        raise SystemExit("release verification failed\n" + "\n".join(failures))
    print(
        json.dumps(
            {
                "status": "PASS",
                "release": manifest["release"],
                "files": len(manifest["files"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
