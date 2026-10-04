"""Regenerate release_manifest.json from the Git-tracked files.

Every tracked file except the manifest itself is recorded with its byte
length and SHA-256, so that ``verify_release.py`` can check a checkout
against the deposited release.

    python build_release_manifest.py --release epoch-manuscript-2026-10-04
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "release_manifest.json"
STUDENT = ROOT / "models" / "epoch_1899_point_student.pt"
MEMORY = ROOT / "models" / "epoch_1899_inference_memory.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True).stdout
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    args = parser.parse_args()

    files = {}
    for relative in tracked_files():
        if relative == MANIFEST.name:
            continue
        path = ROOT / relative
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    manifest = {
        "schema_version": "1.0",
        "release": args.release,
        "knowledge_horizon": 1899,
        "model": {
            "dual_encoder_parameters": 20908547,
            "query_student_parameters": 9868672,
            "query_student_sha256": sha256(STUDENT),
            "inference_memory_sha256": sha256(MEMORY),
        },
        "files": files,
    }
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"release": args.release, "files": len(files)}))


if __name__ == "__main__":
    main()
