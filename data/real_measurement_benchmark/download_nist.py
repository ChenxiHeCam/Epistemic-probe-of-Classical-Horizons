"""Download the manifest-pinned public NIST Dataplot files."""

import hashlib
import json
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
raw = HERE / "raw"
raw.mkdir(exist_ok=True)
base = "https://itl.nist.gov/div898/software/dataplot/data/"
results = []
for row in manifest["datasets"]:
    identifier = row["id"]
    url = base + identifier + ".DAT"
    target = raw / f"{identifier}.DAT"
    status = "downloaded"
    error = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "EPOCH-research-audit/1.0"})
        payload = urllib.request.urlopen(request, timeout=60).read()
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        size = len(payload)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        digest = None
        size = 0
    results.append({"id": identifier, "url": url, "status": status, "bytes": size,
                    "sha256": digest, "error": error})
    print(identifier, status, size)
(HERE / "download_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
