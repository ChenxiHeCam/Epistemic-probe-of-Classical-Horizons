"""Compile the EPOCH manuscript and Supplementary Information.

The .tex files in this directory are the source of record.

  manuscript     NeurIPS 2026 preprint template, pdflatex + bibtex
  supplementary  same template, pdflatex only

Figures are resolved through \graphicspath{{../}} from the repository
``figures/`` directory, so run from a full checkout:

    python paper/build.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# stem, runs bibtex
DOCS = [
    ("manuscript", True),
    ("supplementary", False),
]

AUX = (".aux", ".log", ".bbl", ".blg", ".out")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def build(stem: str, bib: bool) -> bool:
    for suffix in AUX:
        (HERE / (stem + suffix)).unlink(missing_ok=True)
    run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"])
    if bib:
        run(["bibtex", stem])
    run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"])
    run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"])

    pdf = HERE / f"{stem}.pdf"
    log = (HERE / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    errors = [line for line in log.splitlines() if line.startswith("!")]
    undefined = (log.count("LaTeX Warning: Citation")
                 + log.count("LaTeX Warning: Reference"))
    if not pdf.exists():
        print(f"{stem}.pdf -> FAILED")
        for line in errors[:5]:
            print("   ", line)
        return False
    print(f"{stem}.pdf -> OK   errors={len(errors)}  undefined={undefined}")
    for line in errors[:5]:
        print("   ", line)
    return not errors


if __name__ == "__main__":
    sys.exit(0 if all(build(*d) for d in DOCS) else 1)
