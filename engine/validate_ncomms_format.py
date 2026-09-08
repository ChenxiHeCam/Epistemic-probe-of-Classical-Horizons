"""Check the EPOCH manuscript against the chosen Nature Communications envelope.

This mechanical audit follows the journal's current Article guidance: title <=15 words,
abstract <=200 words, main text ideally <=5,000 words, Methods typically
<3,000 words, <=10 display items and figure legends <=350 words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "CLOSURE_PAPER_DRAFT.md"
COVER = ROOT / "manuscript" / "COVER_LETTER.md"
FIGURE = ROOT / "figures" / "figure_case_study_composite.png"
FIGURE_PDF = FIGURE.with_suffix(".pdf")
FIGURE_SVG = FIGURE.with_suffix(".svg")
RESULT = ROOT / "results" / "ncomms_format_audit.json"


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*", text))


def between(text: str, start: str, end: str) -> str:
    match = re.search(rf"(?s){re.escape(start)}\s+(.*?)\s+{re.escape(end)}", text)
    if not match:
        raise AssertionError(f"cannot find section interval {start!r} -> {end!r}")
    return match.group(1)


def main() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    cover = COVER.read_text(encoding="utf-8")
    title = manuscript.splitlines()[0].removeprefix("# ").strip()
    abstract = between(manuscript, "## Abstract", "## Introduction")
    introduction = between(manuscript, "## Introduction", "## Results")
    results = between(manuscript, "## Results", "## Discussion")
    discussion = between(manuscript, "## Discussion", "## Methods")
    methods = between(manuscript, "## Methods", "## Figure legends")
    references = between(manuscript, "## References", "## Data availability")

    title_words = words(title)
    abstract_words = words(abstract)
    main_words = words(introduction + " " + results + " " + discussion)
    methods_words = words(methods)
    reference_numbers = {
        int(number) for number in re.findall(r"(?m)^(\d+)\.\s", references)
    }
    citation_numbers: set[int] = set()
    for group in re.findall(
        r"<sup>([0-9,–-]+)</sup>",
        introduction + results + discussion + methods,
    ):
        for item in group.split(","):
            if "–" in item or "-" in item:
                lo, hi = re.split(r"[–-]", item)
                citation_numbers.update(range(int(lo), int(hi) + 1))
            else:
                citation_numbers.add(int(item))
    image_names = re.findall(r"!\[\]\(figures/([^\)]+)\)", manuscript)
    table_count = len(re.findall(r"(?m)^## Table \d+", manuscript))
    display_items = len(image_names) + table_count
    legends = {
        int(number): words(body)
        for number, body in re.findall(
            r"(?s)\*\*Figure (\d+) \|.*?\*\*(.*?)(?=\n\n!\[\]|\n\n\*Figures|\n\n##)",
            manuscript,
        )
    }

    headings = re.findall(r"(?m)^## (.+)$", manuscript)
    forbidden = [heading for heading in headings if re.match(r"\d+\.", heading)]
    forbidden += [heading for heading in headings if heading in {"Scope", "Additional information"}]

    with Image.open(FIGURE) as image:
        width_px, height_px = image.size
        mode = image.mode
        dpi = image.info.get("dpi", (0, 0))
    width_mm = width_px / float(dpi[0]) * 25.4 if dpi[0] else 0.0

    checks = {
        "title_at_most_15_words": title_words <= 15,
        "abstract_at_most_200_words": abstract_words <= 200,
        "main_text_at_most_5000_words": main_words <= 5000,
        "methods_below_3000_words": methods_words < 3000,
        "display_items_at_most_10": display_items <= 10,
        "figure_legends_at_most_350_words": bool(legends) and max(legends.values()) <= 350,
        "nature_section_names_not_numbered": not forbidden,
        "numbered_references_present": bool(reference_numbers),
        "all_citations_have_references": citation_numbers <= reference_numbers,
        "all_references_are_cited": reference_numbers <= citation_numbers,
        "target_journal_named_in_cover": "Nature Communications" in cover,
        "former_target_absent_from_cover": "Nature Computational Science" not in cover,
        "composite_png_is_rgb": mode == "RGB",
        "composite_png_at_least_300_dpi": min(dpi) >= 300,
        "composite_width_is_180mm": abs(width_mm - 180.0) <= 1.0,
        "vector_masters_exist": FIGURE_PDF.is_file() and FIGURE_SVG.is_file(),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "mechanical Nature Communications format envelope",
        "metrics": {
            "title": title,
            "title_words": title_words,
            "abstract_words": abstract_words,
            "main_text_words": main_words,
            "methods_words": methods_words,
            "references": len(reference_numbers),
            "cited_reference_numbers": sorted(citation_numbers),
            "main_figure_files": image_names,
            "main_figures": len(image_names),
            "main_tables": table_count,
            "display_items": display_items,
            "figure_legend_words": legends,
            "composite_png_pixels": [width_px, height_px],
            "composite_png_mode": mode,
            "composite_png_dpi": list(dpi),
            "composite_width_mm": width_mm,
        },
        "checks": checks,
        "remaining_submission_action": (
            "Insert the permanent versioned data and code accession links before submission."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
