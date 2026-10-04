"""Hash the evidence artifacts into a reproducibility ledger.

This ledger records a retrospective development state.  It is not a protocol
freeze and must not be presented as prospective registration.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "legacy_strict_real_cases": ("engine/strict_evidence_audit.py", "results/strict_evidence.json"),
    "uncapped_diagnostic": ("engine/strict_evidence_audit.py", "results/strict_evidence_uncapped_log.json"),
    "size_matched_generic_detector": ("engine/calibrated_detector_v2.py", "results/calibrated_detector_v2.json"),
    "theory_clock": ("casestudies/theory_clock_experiment.py", "results/theory_clock_experiment.json"),
    "irrelevant_horizon_controls": ("casestudies/irrelevant_horizon_controls.py", "results/irrelevant_horizon_controls.json"),
    "specific_heat_sensitivity": ("casestudies/specific_heat_error_sensitivity.py", "results/specific_heat_error_sensitivity.json"),
    "instrument_robustness": ("casestudies/instrument_robustness.py", "results/instrument_robustness.json"),
    "naive_boundary_bootstrap": ("casestudies/boundary_interval_coverage.py", "results/boundary_interval_coverage.json"),
    "conformal_boundary_intervals": ("casestudies/conformal_boundary_intervals.py", "results/conformal_boundary_intervals.json"),
    "learned_ablation": ("engine/learned_ablation_suite.py", "results/learned_ablation_suite.json"),
    "corpus_provenance": ("engine/audit_classical_corpus_provenance.py", "results/classical_corpus_provenance_audit.json"),
    "real_measurement_inventory": ("data/real_measurement_benchmark/audit_downloads.py", "data/real_measurement_benchmark/inventory_report.json"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    analyses = {}
    for name, (script_name, result_name) in PAIRS.items():
        script = ROOT / script_name
        result = ROOT / result_name
        analyses[name] = {
            "script": script_name,
            "script_sha256": digest(script),
            "result": result_name,
            "result_sha256": digest(result),
        }
    document_files = [
        "protocols/CONFIRMATORY_PROTOCOL_TEMPLATE.md",
    ]
    ledger = {
        "status": "retrospective development evidence ledger; not a preregistration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analyses": analyses,
        "documents": {name: digest(ROOT / name) for name in document_files},
    }
    out = ROOT / "results" / "evidence_manifest.json"
    out.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
