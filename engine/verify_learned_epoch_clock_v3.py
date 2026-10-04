"""Independent arithmetic and hash audit for the EPOCH learned-clock v3 run.

The training/evaluation programs write the scientific results.  This script is
deliberately separate from those programs: it reloads only their frozen JSON
outputs, verifies the SHA-256 sidecars, and recomputes every reported AUROC and
paired clock statistic directly from the deposited group/family scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


EXPECTED = (
    "phase_a_pre1900_v3_evaluation.json",
    "phase_b_pre1950_v3_evaluation.json",
    "epoch_clock_v3_comparison.json",
    "random_phase_a_pre1900_v3_evaluation.json",
    "random_phase_b_pre1950_v3_evaluation.json",
    "random_epoch_clock_v3_comparison.json",
    "matched_pre1900_domain_control_trained_v3.json",
    "matched_pre1900_domain_control_random_v3.json",
)
EXPECTED_MODELS = {
    "phase_a_pre1900_v3_evaluation.json": "phase_a_pre1900_checkpoint_final.pt",
    "phase_b_pre1950_v3_evaluation.json": "phase_b_pre1950_checkpoint_final.pt",
}
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=TOLERANCE):
        raise AssertionError(f"{label}: recomputed {observed}, reported {expected}")


def auroc(negative: list[float], positive: list[float]) -> float:
    """Mann-Whitney AUROC with half credit for exact ties."""
    if not negative or not positive:
        raise AssertionError("AUROC class is empty")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return wins / (len(negative) * len(positive))


def scores(payload: dict, set_name: str, metric: str) -> list[float]:
    rows = payload["sets"][set_name]["scores"]
    return [float(row[metric]) for row in rows.values()]


def comparison_sets(name: str) -> tuple[str, str]:
    mapping = {
        "pre1900_internal_vs_generated_1901_1950": (
            "known_internal", "generated_1901_1950"
        ),
        "known_internal_vs_curated_1901_1950": (
            "known_internal", "curated_1901_1950"
        ),
        "known_internal_vs_curated_all_post1900": (
            "known_internal", "curated_all_post1900"
        ),
        "known_internal_vs_curated_post1950": (
            "known_internal", "curated_post1950"
        ),
        "known_pre1900_internal_vs_curated_post1950": (
            "known_internal_pre1900", "curated_post1950"
        ),
        "known_pre1950_internal_vs_curated_post1950": (
            "known_internal_pre1950", "curated_post1950"
        ),
    }
    if name not in mapping:
        raise AssertionError(f"unregistered comparison in audit: {name}")
    return mapping[name]


def audit_evaluation(payload: dict, label: str) -> dict:
    if payload.get("status") != "PASS":
        raise AssertionError(f"{label}: status is not PASS")
    metric = payload["primary_metric"]
    checked: dict[str, float] = {}
    for name, reported in payload["comparisons"].items():
        negative_set, positive_set = comparison_sets(name)
        negative = scores(payload, negative_set, metric)
        positive = scores(payload, positive_set, metric)
        value = auroc(negative, positive)
        check_close(value, reported["auroc"], f"{label}/{name}")
        if len(negative) != int(reported["negative_groups"]):
            raise AssertionError(f"{label}/{name}: negative group-count mismatch")
        if len(positive) != int(reported["positive_groups"]):
            raise AssertionError(f"{label}/{name}: positive group-count mismatch")
        checked[name] = value

    for name, components in payload["component_aurocs_descriptive"].items():
        negative_set, positive_set = comparison_sets(name)
        for component, reported in components.items():
            value = auroc(
                scores(payload, negative_set, component),
                scores(payload, positive_set, component),
            )
            check_close(value, reported, f"{label}/{name}/{component}")
    return {"comparisons_recomputed": checked}


def exact_sign_test(decreases: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    tail = min(decreases, non_ties - decreases)
    probability = sum(math.comb(non_ties, k) for k in range(tail + 1)) / 2**non_ties
    return min(1.0, 2.0 * probability)


def audit_clock(clock: dict, old: dict, new: dict, label: str) -> dict:
    if clock.get("status") != "PASS":
        raise AssertionError(f"{label}: status is not PASS")
    metric = clock["score"]
    old_rows = old["sets"]["curated_1901_1950"]["scores"]
    new_rows = new["sets"]["curated_1901_1950"]["scores"]
    if set(old_rows) != set(new_rows) or set(old_rows) != set(clock["paired_scores"]):
        raise AssertionError(f"{label}: paired family keys differ")

    old_values: list[float] = []
    new_values: list[float] = []
    collapse: list[float] = []
    for family in sorted(old_rows):
        old_value = float(old_rows[family][metric])
        new_value = float(new_rows[family][metric])
        delta = old_value - new_value
        row = clock["paired_scores"][family]
        check_close(old_value, row["pre1900_score"], f"{label}/{family}/old")
        check_close(new_value, row["pre1950_score"], f"{label}/{family}/new")
        check_close(delta, row["collapse"], f"{label}/{family}/collapse")
        old_values.append(old_value)
        new_values.append(new_value)
        collapse.append(delta)

    decreases = sum(value > 0 for value in collapse)
    increases = sum(value < 0 for value in collapse)
    ties = sum(value == 0 for value in collapse)
    check_close(statistics.median(old_values), clock["pre1900_median"], f"{label}/old median")
    check_close(statistics.median(new_values), clock["pre1950_median"], f"{label}/new median")
    check_close(statistics.median(collapse), clock["median_collapse"], f"{label}/median collapse")
    check_close(statistics.fmean(collapse), clock["mean_collapse"], f"{label}/mean collapse")
    if (decreases, increases, ties) != (
        int(clock["families_decreased"]),
        int(clock["families_increased"]),
        int(clock["ties"]),
    ):
        raise AssertionError(f"{label}: sign counts differ")
    sign_p = exact_sign_test(decreases, decreases + increases)
    check_close(sign_p, clock["exact_two_sided_sign_test_p"], f"{label}/sign test")
    repetitions = int(clock["provenance"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(clock["provenance"]["seed"]))
    collapse_array = np.asarray(collapse, dtype=np.float64)
    boot_median = np.empty(repetitions, dtype=np.float64)
    boot_mean = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = collapse_array[
            rng.integers(0, len(collapse_array), len(collapse_array))
        ]
        boot_median[index] = np.median(sample)
        boot_mean[index] = np.mean(sample)
    for value, reported in zip(
        np.quantile(boot_median, [0.025, 0.975]), clock["median_collapse_ci95"]
    ):
        check_close(value, reported, f"{label}/median bootstrap CI")
    for value, reported in zip(
        np.quantile(boot_mean, [0.025, 0.975]), clock["mean_collapse_ci95"]
    ):
        check_close(value, reported, f"{label}/mean bootstrap CI")
    return {
        "families": len(collapse),
        "median_collapse": statistics.median(collapse),
        "decreases": decreases,
        "increases": increases,
        "ties": ties,
        "exact_two_sided_sign_test_p": sign_p,
    }


def audit_domain(domain: dict, primary: dict, label: str) -> dict:
    if domain.get("status") != "PASS":
        raise AssertionError(f"{label}: status is not PASS")
    metric = domain["primary_metric_reused"]
    controls = scores(domain, "generator_matched_pre1900_controls", metric)
    native = scores(domain, "native_known_internal", metric)
    checked: dict[str, float] = {}
    for name, reported in domain["comparisons"].items():
        if name == "native_internal_vs_matched_pre1900_domain_shift":
            negative, positive = native, controls
        elif name.startswith("matched_pre1900_vs_"):
            positive_set = name.removeprefix("matched_pre1900_vs_")
            negative, positive = controls, scores(primary, positive_set, metric)
        else:
            raise AssertionError(f"{label}: unregistered comparison {name}")
        value = auroc(negative, positive)
        check_close(value, reported["auroc"], f"{label}/{name}")
        checked[name] = value
    for positive_set, components in domain["component_aurocs_descriptive"].items():
        for component, reported in components.items():
            value = auroc(
                scores(domain, "generator_matched_pre1900_controls", component),
                scores(primary, positive_set, component),
            )
            check_close(value, reported, f"{label}/{positive_set}/{component}")
    return {"comparisons_recomputed": checked}


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        raise AssertionError(f"missing sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256(path)
    if observed != expected:
        raise AssertionError(f"hash mismatch: {path.name}")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "learned_epoch_clock_v3_independent_audit.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "PAPER3_ABDUCTION"
        / "EPOCH_MODELS_V3",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    result_dir = args.results_dir.resolve()
    model_dir = args.model_dir.resolve()
    missing = [name for name in EXPECTED if not (result_dir / name).exists()]
    missing_models = [
        name for name in EXPECTED_MODELS.values() if not (model_dir / name).exists()
    ]
    if (missing or missing_models) and not args.allow_incomplete:
        raise FileNotFoundError(
            f"missing frozen outputs={missing}; missing checkpoints={missing_models}"
        )

    present = [name for name in EXPECTED if (result_dir / name).exists()]
    payloads = {name: load(result_dir / name) for name in present}
    hashes = {name: verify_sidecar(result_dir / name) for name in present}
    model_hashes: dict[str, str] = {}
    for result_name, model_name in EXPECTED_MODELS.items():
        model_path = model_dir / model_name
        if not model_path.exists():
            continue
        model_hash = verify_sidecar(model_path)
        model_hashes[model_name] = model_hash
        if result_name in payloads:
            expected_hash = payloads[result_name]["provenance"]["checkpoint_sha256"]
            if model_hash != expected_hash:
                raise AssertionError(
                    f"{model_name}: checkpoint disagrees with {result_name} provenance"
                )
    checks: dict[str, dict] = {}

    for name in (
        "phase_a_pre1900_v3_evaluation.json",
        "phase_b_pre1950_v3_evaluation.json",
        "random_phase_a_pre1900_v3_evaluation.json",
        "random_phase_b_pre1950_v3_evaluation.json",
    ):
        if name in payloads:
            checks[name] = audit_evaluation(payloads[name], name)

    clock_specs = (
        (
            "epoch_clock_v3_comparison.json",
            "phase_a_pre1900_v3_evaluation.json",
            "phase_b_pre1950_v3_evaluation.json",
        ),
        (
            "random_epoch_clock_v3_comparison.json",
            "random_phase_a_pre1900_v3_evaluation.json",
            "random_phase_b_pre1950_v3_evaluation.json",
        ),
    )
    for clock_name, old_name, new_name in clock_specs:
        if all(name in payloads for name in (clock_name, old_name, new_name)):
            checks[clock_name] = audit_clock(
                payloads[clock_name], payloads[old_name], payloads[new_name], clock_name
            )

    domain_specs = (
        (
            "matched_pre1900_domain_control_trained_v3.json",
            "phase_a_pre1900_v3_evaluation.json",
        ),
        (
            "matched_pre1900_domain_control_random_v3.json",
            "random_phase_a_pre1900_v3_evaluation.json",
        ),
    )
    for domain_name, primary_name in domain_specs:
        if domain_name in payloads and primary_name in payloads:
            checks[domain_name] = audit_domain(
                payloads[domain_name], payloads[primary_name], domain_name
            )

    audit = {
        "status": "PASS" if not missing and not missing_models else "PASS_INCOMPLETE",
        "scope": "independent hash and arithmetic recomputation from frozen family/group scores",
        "files_present": present,
        "files_missing": missing,
        "sha256": hashes,
        "checkpoint_files_missing": missing_models,
        "checkpoint_sha256": model_hashes,
        "checks": checks,
        "guardrail": (
            "This audit verifies deposited arithmetic and file identity; it does not "
            "remove benchmark, provenance, generator-domain or external-validity limits."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "checked": list(checks), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
