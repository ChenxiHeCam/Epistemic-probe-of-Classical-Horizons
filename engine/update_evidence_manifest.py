"""Refresh hashes for the Paper 2 evidence ledger.

The ledger records retrospective development assets; it is not a substitute
for the still-missing prospective sealed evaluation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "evidence_manifest.json"

ANALYSES = {
    "legacy_strict_real_cases": (
        "engine/strict_evidence_audit.py",
        "results/strict_evidence.json",
        [],
    ),
    "uncapped_diagnostic": (
        "engine/strict_evidence_audit.py",
        "results/strict_evidence_uncapped_log.json",
        [],
    ),
    "size_matched_generic_detector": (
        "engine/calibrated_detector_v2.py",
        "results/calibrated_detector_v2.json",
        [],
    ),
    "theory_clock": (
        "casestudies/theory_clock_experiment.py",
        "results/theory_clock_experiment.json",
        [],
    ),
    "irrelevant_horizon_controls": (
        "casestudies/irrelevant_horizon_controls.py",
        "results/irrelevant_horizon_controls.json",
        [],
    ),
    "physical_irrelevant_horizon_controls": (
        "casestudies/physical_irrelevant_horizon_controls.py",
        "results/physical_irrelevant_horizon_controls.json",
        [],
    ),
    "specific_heat_sensitivity": (
        "casestudies/specific_heat_error_sensitivity.py",
        "results/specific_heat_error_sensitivity.json",
        [],
    ),
    "instrument_robustness": (
        "casestudies/instrument_robustness.py",
        "results/instrument_robustness.json",
        [],
    ),
    "naive_boundary_bootstrap": (
        "casestudies/boundary_interval_coverage.py",
        "results/boundary_interval_coverage.json",
        [],
    ),
    "conformal_boundary_intervals": (
        "casestudies/conformal_boundary_intervals.py",
        "results/conformal_boundary_intervals.json",
        [],
    ),
    "learned_ablation": (
        "engine/learned_ablation_suite.py",
        "results/learned_ablation_suite.json",
        [],
    ),
    "corpus_provenance": (
        "engine/audit_classical_corpus_provenance.py",
        "results/classical_corpus_provenance_audit.json",
        [],
    ),
    "real_measurement_inventory": (
        "data/real_measurement_benchmark/audit_downloads.py",
        "data/real_measurement_benchmark/inventory_report.json",
        [],
    ),
    "strict_pre1900_registry_reconstruction": (
        "engine/populate_pre1900_registry_from_named_archive.py",
        "results/pre1900_registry_population_report.json",
        ["data/pre1900_registry.json"],
    ),
    "strict_pre1900_pointcloud_subset": (
        "engine/build_audited_pointcloud_subset.py",
        "results/pre1900_audited_pointclouds_manifest.json",
        ["data/pre1900_audited_pointclouds.npz", "data/pre1900_audited_records.jsonl"],
    ),
    "strict_pre1900_encoder_training_with_units": (
        "engine/train_pre1900_strict_encoder.py",
        "results/pre1900_strict_encoder_training_with_units.json",
        ["data/encoder_pre1900_strict_with_units.pt"],
    ),
    "strict_pre1900_encoder_training_zero_units": (
        "engine/train_pre1900_strict_encoder.py",
        "results/pre1900_strict_encoder_training_zero_units.json",
        ["data/encoder_pre1900_strict_zero_units.pt"],
    ),
    "strict_pre1900_one_class_evaluation": (
        "engine/evaluate_pre1900_strict_encoder.py",
        "results/pre1900_strict_one_class.json",
        [],
    ),
    "strict_pre1900_cross_family_conformal": (
        "engine/evaluate_pre1900_cross_family.py",
        "results/pre1900_cross_family_conformal.json",
        [],
    ),
    "strict_pre1900_real_case_scores": (
        "engine/score_real_cases_pre1900_strict.py",
        "results/pre1900_strict_real_case_scores.json",
        [],
    ),
    "matched_generic_function_training": (
        "engine/train_general_function_control.py",
        "results/general_function_control_training.json",
        [
            "data/general_function_control_pointclouds.npz",
            "data/encoder_general_function_control_zero_units.pt",
        ],
    ),
    "matched_generic_function_evaluation": (
        "engine/evaluate_general_function_control.py",
        "results/general_function_control_evaluation.json",
        [],
    ),
    "representation_training_seed_sensitivity": (
        "engine/representation_seed_sensitivity.py",
        "results/representation_seed_sensitivity.json",
        [],
    ),
    "unit_metadata_preserving_deformation_ablation": (
        "engine/evaluate_unit_consistent_deformations.py",
        "results/unit_consistent_deformation_ablation.json",
        [],
    ),
    "post1900_formula_benchmark_build": (
        "engine/build_post1900_formula_benchmark.py",
        "results/post1900_formula_benchmark_manifest.json",
        [
            "data/post1900_formula_registry.json",
            "data/post1900_formula_pointclouds.npz",
            "data/post1900_formula_records.jsonl",
        ],
    ),
    "post1900_formula_benchmark_audit": (
        "engine/audit_post1900_formula_benchmark.py",
        "results/post1900_formula_benchmark_audit.json",
        [],
    ),
    "frozen_post1900_formula_evaluation": (
        "engine/evaluate_frozen_post1900_formulas.py",
        "results/frozen_post1900_formula_evaluation.json",
        [],
    ),
    "case_study_expansion_build": (
        "engine/build_case_study_expansion.py",
        "results/case_study_expansion_manifest.json",
        [
            "data/case_study_expansion_registry.json",
            "data/case_study_expansion_pointclouds.npz",
            "data/case_study_expansion_records.jsonl",
        ],
    ),
    "case_study_expansion_frozen_components": (
        "engine/evaluate_case_study_expansion.py",
        "results/case_study_expansion_evaluation.json",
        [
            "data/statistical_n120_calibration_scores.npz",
            "figures/plot_case_study_composite.py",
        ],
    ),
    "case_study_expansion_large_learned": (
        "../PAPER3_ABDUCTION/evaluate_case_study_expansion_v3.py",
        "results/case_study_expansion_v3_evaluation.json",
        ["../PAPER3_ABDUCTION/EPOCH_RUNTIME_V3/phase_a_inference_memory.pt"],
    ),
    "nature_communications_format_audit": (
        "engine/validate_ncomms_format.py",
        "results/ncomms_format_audit.json",
        [
            "figures/figure_case_study_composite.pdf",
            "figures/figure_case_study_composite.svg",
        ],
    ),
    "strict_pre1900_symbolic_corpus": (
        "../PAPER3_ABDUCTION/audit_epoch_pre1900_v1.py",
        "../PAPER3_ABDUCTION/EPOCH_PRE1900_V1/audit_report.json",
        [
            "../PAPER3_ABDUCTION/build_epoch_pre1900_v1.py",
            "../PAPER3_ABDUCTION/pre1900_source_registry_v1.json",
            "../PAPER3_ABDUCTION/EPOCH_PRE1900_V1/manifest.json",
            "../PAPER3_ABDUCTION/EPOCH_PRE1900_V1/strict_core.jsonl.gz",
            "../PAPER3_ABDUCTION/EPOCH_PRE1900_V1/gold_abduction.jsonl.gz",
        ],
    ),
    "large_scale_epoch_data_readiness": (
        "../PAPER3_ABDUCTION/audit_epoch_training_readiness.py",
        "../PAPER3_ABDUCTION/EPOCH_DATA_READY_V2/readiness_report.json",
        [
            "../PAPER3_ABDUCTION/EPOCH_POINTCLOUD_FREEZE_V2/manifest.json",
            "../PAPER3_ABDUCTION/EPOCH_POINTCLOUD_VIEWS_V2/manifest.json",
            "../PAPER3_ABDUCTION/EPOCH_FORMULA_GRAPHS_V2/manifest.json",
            "../PAPER3_ABDUCTION/EPOCH_RUN_CONFIGS_V2/train_pre1900.json",
            "../PAPER3_ABDUCTION/EPOCH_RUN_CONFIGS_V2/train_pre1950_joint.json",
            "../PAPER3_ABDUCTION/EPOCH_RUN_CONFIGS_V2/test_both.json",
            "../PAPER3_ABDUCTION/EPOCH_POINTCLOUD_ARCHIVE_V2/README.md",
            "../PAPER3_ABDUCTION/EPOCH_POINTCLOUD_ARCHIVE_V2/pointcloud_full_final.tar",
            "../PAPER3_ABDUCTION/EPOCH_POINTCLOUD_ARCHIVE_V2/pointcloud_full_final.tar.sha256",
        ],
    ),
    "large_scale_epoch_phase_a": (
        "../PAPER3_ABDUCTION/evaluate_epoch_horizon_v3.py",
        "results/phase_a_pre1900_v3_evaluation.json",
        [
            "../PAPER3_ABDUCTION/epoch_learned_v3.py",
            "../PAPER3_ABDUCTION/train_epoch_horizon_v3.py",
            "../PAPER3_ABDUCTION/results/phase_a_training_summary.json",
            "../PAPER3_ABDUCTION/EPOCH_MODELS_V3/phase_a_pre1900_checkpoint_final.pt",
            "figures/plot_learned_pre1900_v3.py",
            "figures/plot_learned_architecture_v3.py",
        ],
    ),
    "large_scale_epoch_phase_b_and_clock": (
        "../PAPER3_ABDUCTION/compare_epoch_clock_v3.py",
        "results/phase_b_pre1950_v3_evaluation.json",
        [
            "results/epoch_clock_v3_comparison.json",
            "../PAPER3_ABDUCTION/results/phase_b_training_summary.json",
            "../PAPER3_ABDUCTION/EPOCH_MODELS_V3/phase_b_pre1950_checkpoint_final.pt",
        ],
    ),
    "large_scale_epoch_random_controls": (
        "../PAPER3_ABDUCTION/make_random_epoch_checkpoint_v3.py",
        "results/random_phase_a_pre1900_v3_evaluation.json",
        [
            "results/random_phase_b_pre1950_v3_evaluation.json",
            "results/random_epoch_clock_v3_comparison.json",
        ],
    ),
    "large_scale_epoch_domain_diagnostic": (
        "../PAPER3_ABDUCTION/evaluate_matched_pre1900_clock_controls_v3.py",
        "results/matched_pre1900_domain_control_trained_v3.json",
        [
            "results/matched_pre1900_domain_control_random_v3.json",
            "../PAPER3_ABDUCTION/EPOCH_DOMAIN_CONTROLS_V3/matched_pre1900_controls_v3.npz",
            "../PAPER3_ABDUCTION/EPOCH_DOMAIN_CONTROLS_V3/matched_pre1900_controls_v3_manifest.json",
        ],
    ),
    "large_scale_epoch_independent_audit": (
        "engine/verify_learned_epoch_clock_v3.py",
        "results/learned_epoch_clock_v3_independent_audit.json",
        ["results/learned_epoch_clock_v3_independent_audit.json.sha256"],
    ),
}

DOCUMENTS = [
    "README.md",
    "manuscript/CLOSURE_PAPER_DRAFT.md",
    "manuscript/APPENDIX.md",
    "manuscript/CASE_STUDY_EXPANSION_V1.md",
    "manuscript/REVISION_AUDIT_20260904.md",
    "manuscript/NMI_CHECKLIST.md",
    "manuscript/COVER_LETTER.md",
    "SUBMISSION/README.md",
    "SUBMISSION/build_epoch.py",
    "SUBMISSION/latex/header.tex",
    "SUBMISSION/manuscript.md",
    "SUBMISSION/manuscript.pdf",
    "SUBMISSION/manuscript.docx",
    "SUBMISSION/supplementary.md",
    "SUBMISSION/supplementary.pdf",
    "SUBMISSION/supplementary.docx",
    "SUBMISSION/cover_letter.md",
    "SUBMISSION/cover_letter.pdf",
    "SUBMISSION/cover_letter.docx",
    "protocols/CONFIRMATORY_PROTOCOL_TEMPLATE.md",
    "protocols/HISTORICAL_CASE_TEST_OBJECTS.md",
    "protocols/epoch_test_object.schema.json",
    "protocols/historical_test_objects.json",
    "../PAPER3_ABDUCTION/EPOCH_PRE1900_V1/README.md",
    "../PAPER3_ABDUCTION/UNCONTROLLED_BATCH_20260906.md",
    "../PAPER3_ABDUCTION/GENERATION_BATCH_MANIFEST_TEMPLATE.json",
    "../PAPER3_ABDUCTION/EPOCH_LEARNED_V3_PREREGISTRATION.json",
    "../PAPER3_ABDUCTION/EPOCH_LEARNED_V3_RANDOM_BASELINE_PREREG.json",
    "../PAPER3_ABDUCTION/EPOCH_LEARNED_V3_EVALUATOR_COMPATIBILITY_ADDENDUM.json",
    "../PAPER3_ABDUCTION/EPOCH_LEARNED_V3_DOMAIN_SHIFT_DIAGNOSTIC_REGISTRATION.json",
    "../PAPER3_ABDUCTION/EPOCH_MODELS_V3/README.md",
]

FIGURES = [
    "figures/Figure_1_concept_revised.png",
    "figures/Figure_1_concept_revised.pdf",
    "figures/figure_breakdowns.png",
    "figures/figure_controls.png",
    "figures/figure_rigor.png",
    "figures/figure_limit.png",
    "figures/figure_revision_evidence.png",
    "figures/figure_revision_evidence.pdf",
    "figures/figure_post1900_frozen_transfer.png",
    "figures/figure_post1900_frozen_transfer.pdf",
    "figures/figure_post1900_formula_atlas.png",
    "figures/figure_post1900_formula_atlas.pdf",
    "figures/figure_case_study_composite.png",
    "figures/figure_case_study_composite.pdf",
    "figures/figure_case_study_composite.svg",
    "figures/figure_learned_epoch_clock_v3.png",
    "figures/figure_learned_epoch_clock_v3.pdf",
    "figures/figure_learned_pre1900_v3.png",
    "figures/figure_learned_pre1900_v3.pdf",
    "figures/figure_learned_architecture_v3.png",
    "figures/figure_learned_architecture_v3.pdf",
]

VALIDATORS = [
    "engine/validate_revision_bundle.py",
    "engine/validate_ncomms_format.py",
    "engine/verify_learned_epoch_clock_v3.py",
    "../PAPER3_ABDUCTION/audit_epoch_pre1900_v1.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["status"] = "retrospective development evidence ledger; not a preregistration"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    # Rebuild rather than append so removed analyses cannot survive as stale pins.
    manifest["analyses"] = {}
    for name, (script_name, result_name, artifact_names) in ANALYSES.items():
        script = ROOT / script_name
        result = ROOT / result_name
        missing = [str(path) for path in [script, result] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing evidence asset(s) for {name}: {missing}")
        entry = {
            "script": script_name,
            "script_sha256": sha256(script),
            "result": result_name,
            "result_sha256": sha256(result),
        }
        if artifact_names:
            artifacts = []
            for artifact_name in artifact_names:
                artifact = ROOT / artifact_name
                if not artifact.exists():
                    raise FileNotFoundError(f"missing artifact for {name}: {artifact}")
                artifacts.append({"path": artifact_name, "sha256": sha256(artifact),
                                  "bytes": artifact.stat().st_size})
            entry["artifacts"] = artifacts
        manifest["analyses"][name] = entry

    manifest["documents"] = {
        name: sha256(ROOT / name) for name in DOCUMENTS if (ROOT / name).exists()
    }
    manifest["figures"] = {
        name: sha256(ROOT / name) for name in FIGURES if (ROOT / name).exists()
    }
    manifest["validators"] = {
        Path(name).name: {"path": name, "sha256": sha256(ROOT / name)}
        for name in VALIDATORS if (ROOT / name).exists()
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"analyses": len(manifest["analyses"]),
                      "documents": len(manifest["documents"]),
                      "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
