"""Fail-closed consistency checks for the current EPOCH revision bundle.

This validator does not turn retrospective development results into a sealed
evaluation.  It only prevents manuscript, protocol, figure, model and corpus
assets from silently drifting apart.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, *, atol: float = 1e-12) -> None:
    assert math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol), (actual, expected)


def validate_ledger() -> dict:
    ledger = load(RESULTS / "evidence_manifest.json")
    assert ledger["status"] == "retrospective development evidence ledger; not a preregistration"
    for name, entry in ledger["analyses"].items():
        script = ROOT / entry["script"]
        result = ROOT / entry["result"]
        assert script.is_file(), (name, script)
        assert result.is_file(), (name, result)
        assert sha256(script) == entry["script_sha256"], (name, "script")
        assert sha256(result) == entry["result_sha256"], (name, "result")
        for artifact in entry.get("artifacts", []):
            path = ROOT / artifact["path"]
            assert path.is_file(), (name, path)
            assert path.stat().st_size == artifact["bytes"], (name, path, "bytes")
            assert sha256(path) == artifact["sha256"], (name, path, "sha256")
    for collection in ("documents", "figures", "validators"):
        for name, pinned in ledger.get(collection, {}).items():
            if isinstance(pinned, dict):
                path = ROOT / pinned["path"]
                expected = pinned["sha256"]
            else:
                path = ROOT / name
                expected = pinned
            assert path.is_file(), (collection, path)
            assert sha256(path) == expected, (collection, path)
    return ledger


def validate_test_objects() -> int:
    protocol = ROOT / "protocols"
    schema = load(protocol / "epoch_test_object.schema.json")
    payload = load(protocol / "historical_test_objects.json")
    objects = payload if isinstance(payload, list) else payload.get("test_objects", payload.get("cases"))
    assert isinstance(objects, list)
    validator = jsonschema.Draft202012Validator(schema)
    errors = [(index, error.json_path, error.message)
              for index, obj in enumerate(objects) for error in validator.iter_errors(obj)]
    assert not errors, errors[:10]
    assert len(objects) == 5
    return len(objects)


def validate_numerical_claims() -> None:
    detector = load(RESULTS / "calibrated_detector_v2.json")
    op = detector["synthetic_mechanism_benchmark"]["operating_points"]
    close(op["5pct"]["tpr"], 0.2708333333333333)
    close(op["5pct"]["fpr"], 0.0)
    close(op["1pct"]["tpr"], 0.0)
    close(op["1pct"]["fpr"], 0.0)
    assert sum(row["reject_bh_5pct"] for row in detector["real_cases"]) == 0

    clock = load(RESULTS / "theory_clock_experiment.json")
    assert len(clock["cases"]) == 5
    assert sum(row["paired_success"] for row in clock["cases"]) == 4

    physical_controls = load(RESULTS / "physical_irrelevant_horizon_controls.json")
    assert len(physical_controls["cases"]) == 5
    assert [row["case"] for row in physical_controls["cases"]] == [
        "FIRAS", "Bertozzi", "specific_heat_Cu", "Onnes", "Millikan"
    ]
    assert all(row["bootstrap_valid"] == 1000 for row in physical_controls["cases"])
    assert all(row["remains_rejected_fdr5"] for row in physical_controls["cases"])
    assert all(row["dimension_audit"] for row in physical_controls["cases"])
    assert all(math.isclose(row["pvalue"], 1 / 1001) for row in physical_controls["cases"])
    assert all(math.isclose(row["bh_qvalue"], 1 / 1001) for row in physical_controls["cases"])

    intervals = load(RESULTS / "conformal_boundary_intervals.json")
    assert intervals["calibration_datasets"] == 899
    assert intervals["test_datasets"] == 900
    close(intervals["overall"]["coverage"], 0.9133333333333333)

    learned = load(RESULTS / "pre1900_strict_one_class.json")
    zero = learned["variants"]["zero_units"]["synthetic"]
    random = learned["variants"]["random_zero_units"]["synthetic"]
    close(zero["knn"]["auroc"], 0.8544921875)
    close(zero["shrinkage_mahalanobis"]["auroc"], 0.8953993055555555)
    close(random["knn"]["auroc"], 0.3291015625)
    close(random["shrinkage_mahalanobis"]["auroc"], 0.28765190972222227)
    assert learned["audited_reference"]["records"] == 268
    assert learned["audited_reference"]["families"] == 51

    grouped = load(RESULTS / "pre1900_cross_family_conformal.json")
    close(grouped["synthetic"]["family_auroc"], 0.9027777777777778)
    close(grouped["synthetic"]["breakdown_family_tpr_at_threshold"], 2 / 12)
    close(grouped["synthetic"]["classical_family_fpr_at_threshold"], 0.0)
    close(grouped["heldout_classical"]["test"]["family_fpr"], 0.0)

    generic = load(RESULTS / "general_function_control_evaluation.json")
    assert generic["protocol"]["record_count_match"]
    assert generic["protocol"]["family_size_profile_match"]
    assert generic["protocol"]["function_signature_histogram_match"]
    close(generic["synthetic"]["family_knn_auroc"], 0.375)
    close(generic["synthetic"]["breakdown_family_tpr_at_threshold"], 10 / 12)
    close(generic["synthetic"]["classical_family_fpr_at_threshold"], 11 / 12)
    close(generic["heldout_cited_pre1900"]["test"]["family_fpr"], 4 / 7)
    close(generic["incremental_comparison"]["strict_minus_generic_family_auroc"], 19 / 36)

    seeds = load(RESULTS / "representation_seed_sensitivity.json")
    assert len(seeds["runs"]) == 3
    strict_seed_auc = [row["strict"]["family_auroc"] for row in seeds["runs"]]
    generic_seed_auc = [row["generic"]["family_auroc"] for row in seeds["runs"]]
    strict_seed_tpr = [row["strict"]["breakdown_family_tpr"] for row in seeds["runs"]]
    assert min(strict_seed_auc) >= 0.888 and max(strict_seed_auc) <= 0.911
    assert min(generic_seed_auc) >= 0.291 and max(generic_seed_auc) <= 0.404
    assert min(strict_seed_tpr) == 1 / 12 and max(strict_seed_tpr) == 8 / 12
    assert all(row["strict"]["synthetic_classical_family_fpr"] == 0 for row in seeds["runs"])
    assert all(row["strict"]["cited_test_family_fpr"] == 0 for row in seeds["runs"])

    units = load(RESULTS / "unit_consistent_deformation_ablation.json")
    assert units["heldout"]["records"] == 46
    assert units["heldout"]["source_law_families"] == 7
    assert units["heldout"]["records_with_any_nonzero_dimension_metadata"] == 34
    assert len(units["heldout"]["deformation_families"]) == 6
    close(units["variants"]["with_units"]["source_family_balanced_auroc"], 0.6836734693877551)
    close(units["variants"]["zero_units"]["source_family_balanced_auroc"], 0.9217687074829931)
    close(units["with_minus_zero_source_family_balanced_auroc"], -0.23809523809523803)
    assert units["paired_source_family_bootstrap_95_ci"][1] < 0
    close(units["variants"]["with_units"]["clean_source_family_fpr"], 0.0)
    close(units["variants"]["zero_units"]["clean_source_family_fpr"], 0.0)
    close(units["variants"]["with_units"]["deformed_source_family_cell_tpr"], 0.0)
    close(units["variants"]["zero_units"]["deformed_source_family_cell_tpr"], 4 / 42)
    for variant in units["variants"].values():
        assert len(variant["calibration_source_family_scores"]) == 40
        assert len(variant["clean_source_family_scores"]) == 7
        assert set(variant["deformed_source_family_scores"]) == set(
            units["heldout"]["deformation_families"]
        )
        assert all(len(scores) == 7 for scores in variant["deformed_source_family_scores"].values())

    post_manifest = load(RESULTS / "post1900_formula_benchmark_manifest.json")
    post_audit = load(RESULTS / "post1900_formula_benchmark_audit.json")
    post = load(RESULTS / "frozen_post1900_formula_evaluation.json")
    assert post_manifest["status"] == "PASS"
    assert post_manifest["families"] == 55
    assert post_manifest["records"] == 440
    assert post_manifest["records_per_family"] == 8
    assert post_manifest["points_per_record"] == 120
    assert post_manifest["year_min"] == 1901 and post_manifest["year_max"] == 1996
    assert post_audit["status"] == "PASS"
    assert post_audit["strict_post1900"]
    assert post_audit["citation_year_matches_declared_year"] == 55
    assert post_audit["citations_with_doi"] == 45
    assert post_audit["family_identifier_overlap_with_pre1900_archive"] == []
    assert len(post_audit["declared_dictionary_overlap_hard_cases"]) == 12
    assert post["status"] == "completed frozen forward-inference audit"
    assert post["no_training_audit"] == {
        "training_steps": 0,
        "optimizer_constructed": False,
        "model_eval_mode": True,
        "weights_updated": False,
        "thresholds_updated": False,
        "post1900_records_entered_training_or_calibration": False,
    }
    primary = post["primary_family_level"]
    close(primary["learned"]["auroc_post1900_vs_heldout_pre1900"], 0.6694214876033058)
    close(primary["statistical_four_form"]["auroc_post1900_vs_heldout_pre1900"], 0.8876033057851239)
    assert primary["learned"]["post1900_alerts"] == 7
    assert primary["statistical_four_form"]["post1900_alerts"] == 7
    assert primary["statistical_four_form"]["q99_post1900_alerts"] == 0
    assert primary["framework_or"]["post1900_alerts"] == 11
    assert primary["framework_and"]["post1900_alerts"] == 3
    for component in ("learned", "statistical_four_form", "framework_or", "framework_and"):
        assert primary[component]["heldout_pre1900_false_alerts"] == 0
        assert primary[component]["heldout_pre1900_families"] == 11
    overlap = post["by_shape_class"]["framework_or_alert_q95"]["declared_dictionary_overlap"]
    assert overlap["families"] == 12 and overlap["alerts"] == 1
    close(overlap["rate"], 1 / 12)
    assert post["primary_family_level"]["statistical_four_form"]["post1900_record_abstentions"] == 2

    expansion_manifest = load(RESULTS / "case_study_expansion_manifest.json")
    expansion = load(RESULTS / "case_study_expansion_evaluation.json")
    expansion_v3 = load(RESULTS / "case_study_expansion_v3_evaluation.json")
    assert expansion_manifest["status"] == "PASS"
    assert expansion_manifest["families"] == 11 and expansion_manifest["records"] == 88
    assert expansion_manifest["original_benchmark_mutated"] is False
    assert expansion["benchmark"]["families"] == 11
    assert expansion["benchmark"]["statistical_calibration_scores"] == 785
    assert expansion["no_training_audit"]["original_55_family_result_mutated"] is False
    assert expansion_v3["status"] == "PASS"
    assert expansion_v3["no_training_audit"]["optimizer_constructed"] is False
    assert expansion_v3["no_training_audit"]["original_55_family_benchmark_mutated"] is False
    assert expansion_v3["reconstruction_check"]["families_compared"] == 38
    assert expansion_v3["reconstruction_check"]["maximum_absolute_ensemble_delta"] <= 0.03
    expansion_large = {row["family_id"]: row["one_class_ensemble"] for row in expansion_v3["case_rows"]}
    expansion_small = {
        row["family_id"]: row["statistical_percentile_vs_785_size_matched_calibration"]
        for row in expansion["case_rows"]
    }
    close(expansion_large["casimir_parallel_plates"], 1.0)
    close(expansion_large["pound_rebka_redshift"], 0.07065217391304347)
    close(expansion_small["cherenkov_threshold"], 0.9650127226463104)
    close(expansion_small["gw150914_leading_chirp"], 0.2741730279898219)

    phase_a = load(RESULTS / "phase_a_pre1900_v3_evaluation.json")
    phase_b = load(RESULTS / "phase_b_pre1950_v3_evaluation.json")
    random_a = load(RESULTS / "random_phase_a_pre1900_v3_evaluation.json")
    random_b = load(RESULTS / "random_phase_b_pre1950_v3_evaluation.json")
    learned_clock = load(RESULTS / "epoch_clock_v3_comparison.json")
    random_clock = load(RESULTS / "random_epoch_clock_v3_comparison.json")
    domain = load(RESULTS / "matched_pre1900_domain_control_trained_v3.json")
    random_domain = load(RESULTS / "matched_pre1900_domain_control_random_v3.json")
    independent = load(RESULTS / "learned_epoch_clock_v3_independent_audit.json")
    assert phase_a["status"] == phase_b["status"] == "PASS"
    assert phase_a["knowledge_cutoff"] == 1899 and phase_b["knowledge_cutoff"] == 1950
    close(
        phase_a["comparisons"]["known_internal_vs_curated_1901_1950"]["auroc"],
        0.9210526315789473,
    )
    close(
        phase_a["comparisons"]["known_internal_vs_curated_all_post1900"]["auroc"],
        0.9252525252525252,
    )
    close(
        phase_a["comparisons"]["known_internal_vs_curated_post1950"]["auroc"],
        0.934640522875817,
    )
    close(
        phase_a["comparisons"]["pre1900_internal_vs_generated_1901_1950"]["auroc"],
        0.5357496533967122,
    )
    close(
        random_a["comparisons"]["known_internal_vs_curated_all_post1900"]["auroc"],
        0.6494949494949495,
    )
    close(
        domain["comparisons"]["matched_pre1900_vs_curated_all_post1900"]["auroc"],
        0.8565656565656565,
    )
    close(
        random_domain["comparisons"]["matched_pre1900_vs_curated_all_post1900"]["auroc"],
        0.5454545454545454,
    )
    selected_case_linked_families = {
        "special_relativity": (0.9320652173913043, 0.9157608695652174),
        "photoelectric_threshold": (0.7934782608695653, 0.09782608695652174),
        "einstein_solid": (0.8505434782608696, 1.0),
        "debye_solid": (0.9157608695652174, 0.9456521739130435),
        "london_two_fluid_penetration": (0.8097826086956521, 0.9701086956521738),
        "ginzburg_landau_order": (0.9619565217391304, 0.7119565217391305),
    }
    trained_families = phase_a["sets"]["curated_all_post1900"]["scores"]
    random_families = random_a["sets"]["curated_all_post1900"]["scores"]
    for family, (trained_score, random_score) in selected_case_linked_families.items():
        close(trained_families[family]["one_class_ensemble"], trained_score)
        close(random_families[family]["one_class_ensemble"], random_score)
    close(
        phase_b["comparisons"]["known_internal_vs_curated_post1950"]["auroc"],
        0.4530360531309298,
    )
    close(
        random_b["comparisons"]["known_internal_vs_curated_post1950"]["auroc"],
        0.7476280834914611,
    )
    close(learned_clock["median_collapse"], 0.42920893719806763)
    assert learned_clock["families_decreased"] == 25
    close(learned_clock["exact_two_sided_sign_test_p"], 0.07295138851623051)
    close(random_clock["median_collapse"], 0.00858695652173913)
    assert independent["status"] == "PASS"
    assert not independent["files_missing"]
    assert not independent["checkpoint_files_missing"]
    assert len(independent["files_present"]) == 8
    model_dir = WORKSPACE / "PAPER3_ABDUCTION" / "EPOCH_MODELS_V3"
    assert sha256(model_dir / "phase_a_pre1900_checkpoint_final.pt") == (
        "3a350d85a79dd02d426d610cc96408b3041485d820e4a3b424da2ba4d00622d0"
    )
    assert sha256(model_dir / "phase_b_pre1950_checkpoint_final.pt") == (
        "ca64446d95bb0966a630412ec60ea5e43461974f74dcc5f8e9cccd16ab57dde5"
    )

    corpus = load(WORKSPACE / "PAPER3_ABDUCTION" / "EPOCH_PRE1900_V1" / "audit_report.json")
    assert corpus["status"] == "PASS"
    assert corpus["files"]["strict_core.jsonl.gz"]["rows"] == 12783
    assert corpus["files"]["gold_abduction.jsonl.gz"]["rows"] == 128
    assert corpus["paper2_pointcloud_bridge"]["records"] == 268
    assert corpus["paper2_pointcloud_bridge"]["families"] == 51


def validate_claim_text() -> None:
    current = [
        ROOT / "README.md",
        ROOT / "manuscript" / "CLOSURE_PAPER_DRAFT.md",
        ROOT / "manuscript" / "APPENDIX.md",
        ROOT / "manuscript" / "REVISION_AUDIT_20260904.md",
        ROOT / "manuscript" / "NMI_CHECKLIST.md",
        ROOT / "manuscript" / "COVER_LETTER.md",
        WORKSPACE / "PAPER3_ABDUCTION" / "EPOCH_PRE1900_V1" / "README.md",
    ]
    stale = (
        "253 point clouds", "253-record", "193 source-audited", "193 training records",
        "11,343 tasks", "50,645 legacy", "family AUROC 0.778", "7/12 breakdown families",
    )
    for path in current:
        text = path.read_text(encoding="utf-8")
        hits = [token for token in stale if token in text]
        assert not hits, (path, hits)
    manuscript = current[1].read_text(encoding="utf-8")
    for required in (
        "model incompatibility over the sampled range, boundary not identifiable",
        "detecting 2/12 breakdown families",
        "268 point clouds from 51 cited pre-1900 law families",
        "without averaging unlike quantities",
        "paired difference of 0.528 [0.264, 0.764]",
        "source-family-balanced AUROC 0.684 with units and 0.922 without them",
        "dimensionally coherent but mechanism-irrelevant forms",
        "citation-anchored temporal-transfer audit containing 55 named physical-law families",
        "learned component ranks post-1900 families above held-out pre-1900 controls with AUROC 0.669",
        "four-form component reaches 0.888 [0.800, 0.959]",
        "AUROC directly measures the frozen ordering of later families against earlier holdouts",
        "the 1899-horizon score reaches AUROC 0.921 [0.833, 0.994] on 38 citation-anchored 1901--1950 families",
        "native generator-gated comparison is near chance (AUROC 0.536)",
        "11 additional twentieth-century cases",
        "including the Lamb shift and electron anomalous magnetic moment",
        "cumulative pre-1950 corpus awaits the same source-level audit",
        "## References",
        "## Limitations and outlook",
    ):
        assert required in manuscript, required
    assert "figure_learned_pre1900_v3.png" in manuscript
    assert "figure_learned_epoch_clock_v3.png" not in manuscript
    assert "figure_case_study_composite.png" in manuscript
    assert "figure_post1900_frozen_transfer.png" not in manuscript
    assert "figure_post1900_formula_atlas.png" not in manuscript
    appendix = current[2].read_text(encoding="utf-8")
    assert "Their BH q-values are 0.107, 0.107, 0.346 and 0.346" in appendix
    assert "The component pattern is itself informative" in appendix
    assert "Archived pre-1950 extension pilot (not a current endpoint)" in appendix
    assert "This outcome-aware rescoping is disclosed explicitly" in appendix
    assert "Post-score case-study expansion and composite visualization" in appendix
    submission = ROOT / "SUBMISSION"
    submission_readme = (submission / "README.md").read_text(encoding="utf-8")
    assert "synchronized development submission package" in submission_readme
    assert "source-audited 1899 knowledge horizon" in submission_readme
    for source, copy in (
        (ROOT / "manuscript" / "CLOSURE_PAPER_DRAFT.md", submission / "manuscript.md"),
        (ROOT / "manuscript" / "APPENDIX.md", submission / "supplementary.md"),
        (ROOT / "manuscript" / "COVER_LETTER.md", submission / "cover_letter.md"),
    ):
        assert source.read_bytes() == copy.read_bytes(), (source, copy, "not synchronized")
    for deliverable in (
        "manuscript.pdf", "manuscript.docx", "supplementary.pdf",
        "supplementary.docx", "cover_letter.pdf", "cover_letter.docx",
    ):
        path = submission / deliverable
        assert path.is_file() and path.stat().st_size > 10_000, path
    submitted_text = (submission / "manuscript.md").read_text(encoding="utf-8")
    assert "0/4 at FDR 5%" not in submitted_text
    figure_names = re.findall(r"!\[\]\(figures/([^\)]+)\)", submitted_text)
    assert len(figure_names) == 8 and len(set(figure_names)) == 8, figure_names
    for figure_name in figure_names:
        source_figure = ROOT / "figures" / figure_name
        submitted_figure = submission / "figures" / figure_name
        assert source_figure.is_file() and submitted_figure.is_file(), figure_name
        assert source_figure.read_bytes() == submitted_figure.read_bytes(), figure_name
    for unsupported in (
        "one unmodified pipeline to flag 4/4 foundational breakdowns",
        "The frozen pipeline flags 4/4 canonical breakdowns",
        "strictly leakage-free set-transformer encoder",
        "cumulative pre-1950 model does not generalize onward",
        "failed forward-transfer result",
    ):
        assert unsupported not in submitted_text, unsupported


def validate_no_embedded_credentials() -> None:
    assignment = re.compile(
        r"(?m)^(?:KEY|API_KEY|TOKEN)\s*=\s*['\"][^'\"]{12,}['\"]\s*$"
    )
    command_argument = re.compile(r"GEN_API_KEY\s*=\s*[A-Za-z0-9_-]{20,}")
    for name in ("glm_gen.py", "glm_repair.py"):
        path = WORKSPACE / "PAPER3_ABDUCTION" / name
        text = path.read_text(encoding="utf-8")
        assert not assignment.search(text), path
        assert "os.environ.get('GEN_API_KEY')" in text, path
        assert "require_batch_authorization" in text, path
        assert "GEN_BATCH_MANIFEST" in text, path
    for base in (ROOT, WORKSPACE / "PAPER3_ABDUCTION"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".ps1", ".sh"}:
                assert not command_argument.search(path.read_text(encoding="utf-8", errors="replace")), path
    incident = (WORKSPACE / "PAPER3_ABDUCTION" / "UNCONTROLLED_BATCH_20260906.md").read_text(
        encoding="utf-8"
    )
    assert "q062–q069" in incident
    assert "must be revoked or rotated" in incident
    template = load(WORKSPACE / "PAPER3_ABDUCTION" / "GENERATION_BATCH_MANIFEST_TEMPLATE.json")
    assert template["status"] == "TEMPLATE_NOT_AUTHORIZED"
    assert template["allowed_waves"] == []


def main() -> None:
    ledger = validate_ledger()
    object_count = validate_test_objects()
    validate_numerical_claims()
    validate_claim_text()
    validate_no_embedded_credentials()
    print(json.dumps({
        "status": "PASS",
        "analyses": len(ledger["analyses"]),
        "documents": len(ledger["documents"]),
        "figures": len(ledger["figures"]),
        "test_objects": object_count,
        "scope": "revision consistency only; prospective sealed evidence remains external",
    }, indent=2))


if __name__ == "__main__":
    main()
