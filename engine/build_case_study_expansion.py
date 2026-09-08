"""Build a post-score formula/simulator case-study expansion for EPOCH.

The original 55-family temporal benchmark remains byte-for-byte untouched.
This development set was selected after those scores were opened and therefore
cannot enlarge the primary AUROC denominator.  It is used to exercise routing,
show learned/statistical complementarity, and design a compact case-study
figure.  Every record is an explicitly labelled formula reduction or stylized
simulator, never a claim of being raw historical measurement data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 20260908
REPEATS = 8
POINTS = 120


CASES = [
    dict(
        family_id="franck_hertz_excitation", short_name="Franck--Hertz",
        law_name="Franck--Hertz discrete-excitation current structure",
        first_valid_year=1914, domain="atomic_physics", generator="franck_hertz",
        equation="I(V) = smooth baseline with instrument-broadened losses near V = n DeltaV",
        x_range=[0.0, 30.0], sampling="linear", shape_class="repeated_thresholds",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="smooth continuous-loss baseline",
        reduction_status="stylized instrument-convolved mechanism simulator; not digitized historical data",
        citation="J. Franck and G. Hertz, Verh. Dtsch. Phys. Ges. 16, 457 and 512 (1914).",
        source_url="https://onlinelibrary.wiley.com/doi/10.1002/phbl.19670230702",
    ),
    dict(
        family_id="bec_condensate_fraction", short_name="BEC fraction",
        law_name="Ideal Bose--Einstein condensate fraction",
        first_valid_year=1925, domain="quantum_statistics", generator="bec_fraction",
        equation="N0/N = max(1-(T/Tc)^(3/2), 0)",
        x_range=[0.05, 1.35], sampling="linear", shape_class="threshold_or_piecewise",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="no macroscopic condensate fraction",
        reduction_status="homogeneous ideal-gas formula stress test; not a trapped-gas measurement",
        citation="A. Einstein, Quantentheorie des einatomigen idealen Gases, Zweite Abhandlung (1925); experimental anchor M. H. Anderson et al., Science 269, 198 (1995).",
        source_url="https://www.science.org/doi/10.1126/science.269.5221.198",
    ),
    dict(
        family_id="cherenkov_threshold", short_name="Cherenkov threshold",
        law_name="Frank--Tamm Cherenkov threshold factor",
        first_valid_year=1937, domain="electrodynamics", generator="cherenkov_threshold",
        equation="Y(beta) proportional to max(1-1/(n^2 beta^2), 0)",
        x_range=[0.40, 0.999], sampling="linear", shape_class="threshold_or_piecewise",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="zero yield below the admitted radiation mechanism",
        reduction_status="fixed-frequency nondispersive threshold factor",
        citation="I. Frank and I. Tamm, Coherent visible radiation of fast electrons passing through matter, C. R. Acad. Sci. USSR 14, 109--114 (1937).",
        source_url="https://cds.cern.ch/record/485596",
    ),
    dict(
        family_id="casimir_parallel_plates", short_name="Casimir plates",
        law_name="Casimir pressure between parallel conducting plates",
        first_valid_year=1948, domain="quantum_field_theory", generator="casimir_pressure",
        equation="P(a) proportional to -a^(-4)",
        x_range=[0.20, 3.0], sampling="log", shape_class="declared_dictionary_overlap",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="zero force under the declared no-vacuum-force null",
        reduction_status="ideal perfect-conductor zero-temperature formula",
        citation="H. B. G. Casimir, Proc. K. Ned. Akad. Wet. 51, 793--795 (1948).",
        source_url="https://dwc.knaw.nl/DL/publications/PU00018547.pdf",
    ),
    dict(
        family_id="parity_beta_asymmetry", short_name="Parity asymmetry",
        law_name="Polarized beta-decay angular asymmetry",
        first_valid_year=1957, domain="weak_interactions", generator="parity_asymmetry",
        equation="W(theta) = 1 + A P cos(theta)",
        x_range=[0.0, float(np.pi)], sampling="linear", shape_class="smooth_composite",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="fore--aft symmetric angular distribution",
        reduction_status="ideal acceptance-free angular law; not Wu event-level data",
        citation="C. S. Wu et al., Phys. Rev. 105, 1413--1415 (1957).",
        source_url="https://journals.aps.org/pr/abstract/10.1103/PhysRev.105.1413",
    ),
    dict(
        family_id="anderson_localized_envelope", short_name="Anderson envelope",
        law_name="Idealized Anderson-localized state envelope",
        first_valid_year=1958, domain="condensed_matter", generator="anderson_envelope",
        equation="|psi(x)|^2 proportional to exp(-2|x|/xi)",
        x_range=[-5.0, 5.0], sampling="linear", shape_class="nonmonotone_or_oscillatory",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="extended-state comparator",
        reduction_status="phenomenological localized-envelope simulator, not a mobility-edge solution",
        citation="P. W. Anderson, Phys. Rev. 109, 1492--1505 (1958).",
        source_url="https://journals.aps.org/pr/abstract/10.1103/PhysRev.109.1492",
    ),
    dict(
        family_id="pound_rebka_redshift", short_name="Pound--Rebka",
        law_name="Weak-field gravitational redshift",
        first_valid_year=1960, domain="gravitation", generator="pound_rebka",
        equation="Delta nu/nu = g h/c^2",
        x_range=[0.0, 30.0], sampling="linear", shape_class="declared_dictionary_overlap",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="zero frequency shift",
        reduction_status="dimensionless weak-field law with rescaled coefficient",
        citation="R. V. Pound and G. A. Rebka Jr., Phys. Rev. Lett. 4, 337--341 (1960).",
        source_url="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.4.337",
    ),
    dict(
        family_id="fano_asymmetric_resonance", short_name="Fano resonance",
        law_name="Fano asymmetric resonance profile",
        first_valid_year=1961, domain="atomic_physics", generator="fano_profile",
        equation="I(epsilon) proportional to (q+epsilon)^2/(1+epsilon^2)",
        x_range=[-5.0, 5.0], sampling="linear", shape_class="nonmonotone_or_oscillatory",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="symmetric Lorentzian comparator",
        reduction_status="canonical Fano profile",
        citation="U. Fano, Phys. Rev. 124, 1866--1878 (1961).",
        source_url="https://journals.aps.org/pr/abstract/10.1103/PhysRev.124.1866",
    ),
    dict(
        family_id="aspect_polarization_correlation", short_name="Bell--Aspect",
        law_name="Quantum polarization correlation in a Bell test",
        first_valid_year=1982, domain="quantum_foundations", generator="aspect_correlation",
        equation="E(theta) = -V cos(2 theta)",
        x_range=[0.0, float(np.pi)], sampling="linear", shape_class="nonmonotone_or_oscillatory",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="local-bound-limited sinusoidal visibility comparator",
        reduction_status="ideal correlation curve; inequality inference requires trial counts",
        citation="A. Aspect, J. Dalibard and G. Roger, Phys. Rev. Lett. 49, 1804--1807 (1982).",
        source_url="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.49.1804",
    ),
    dict(
        family_id="accelerating_universe_residual", short_name="Cosmic acceleration",
        law_name="Flat LambdaCDM distance-modulus residual over Einstein--de Sitter",
        first_valid_year=1998, domain="cosmology", generator="accelerating_universe",
        equation="Delta mu(z) = 5 log10[dL(0.3,0.7)/dL(1,0)]",
        x_range=[0.01, 1.20], sampling="linear", shape_class="smooth_composite",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="zero residual under the declared Omega_M=1, Omega_Lambda=0 curve",
        reduction_status="ideal homogeneous-cosmology curve; real-data inference needs covariance and selection effects",
        citation="A. G. Riess et al., Astron. J. 116, 1009--1038 (1998).",
        source_url="https://arxiv.org/abs/astro-ph/9805201",
    ),
    dict(
        family_id="gw150914_leading_chirp", short_name="GW chirp",
        law_name="Leading inspiral gravitational-wave chirp",
        first_valid_year=2016, domain="gravitation", generator="gw_chirp",
        equation="f(tau) proportional to tau^(-3/8)",
        x_range=[0.08, 8.0], sampling="log", shape_class="declared_dictionary_overlap",
        eligible_components=["learned", "data_anchored", "theory_conditioned"],
        incumbent="stationary-frequency comparator",
        reduction_status="leading-order chirp reduction; not detector strain or a full GR waveform",
        citation="B. P. Abbott et al., Phys. Rev. Lett. 116, 061102 (2016).",
        source_url="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102",
    ),
]


THEORY_ONLY = [
    {
        "case_id": "lamb_shift", "year": 1947,
        "reason_not_pointcloud": "the decisive historical object is a sparse nonzero level splitting/degeneracy test",
        "route": "theory_conditioned",
        "source_url": "https://journals.aps.org/pr/abstract/10.1103/PhysRev.72.241",
    },
    {
        "case_id": "electron_anomalous_magnetic_moment", "year": 1948,
        "reason_not_pointcloud": "the decisive object is a coefficient/scalar departure from Dirac g=2",
        "route": "theory_conditioned",
        "source_url": "https://journals.aps.org/pr/abstract/10.1103/PhysRev.73.416",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_x(row: dict, rng: np.random.Generator) -> np.ndarray:
    lo, hi = map(float, row["x_range"])
    x = np.geomspace(lo, hi, POINTS) if row["sampling"] == "log" else np.linspace(lo, hi, POINTS)
    local = np.r_[np.diff(x)[0], (np.diff(x)[:-1] + np.diff(x)[1:]) / 2, np.diff(x)[-1]]
    x = x + rng.normal(0, 0.08, POINTS) * local
    return np.sort(np.clip(x, lo, hi))


def luminosity_distance_flat(z: np.ndarray, omega_m: float, omega_l: float) -> np.ndarray:
    grid = np.r_[0.0, np.asarray(z, float)]
    ez = np.sqrt(omega_m * (1 + grid) ** 3 + omega_l)
    integral = cumulative_trapezoid(1 / ez, grid, initial=0.0)
    return (1 + grid[1:]) * integral[1:]


def evaluate(generator: str, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    amplitude = float(rng.uniform(0.85, 1.15))
    if generator == "franck_hertz":
        spacing = float(rng.uniform(4.7, 5.1)); width = float(rng.uniform(0.28, 0.48))
        baseline = 0.45 + 0.075 * x
        loss = np.zeros_like(x)
        for centre in np.arange(spacing, x.max() + spacing, spacing):
            loss += np.exp(-0.5 * ((x - centre) / width) ** 2)
        depth = float(rng.uniform(0.28, 0.42))
        y = amplitude * baseline * np.maximum(1 - depth * loss, 0.08)
        old = amplitude * baseline
        parameters = {"amplitude": amplitude, "spacing": spacing, "width": width, "depth": depth}
    elif generator == "bec_fraction":
        critical = float(rng.uniform(0.94, 1.06))
        y = amplitude * np.maximum(1 - (x / critical) ** 1.5, 0)
        old = np.zeros_like(x)
        parameters = {"amplitude": amplitude, "critical": critical}
    elif generator == "cherenkov_threshold":
        refractive_index = float(rng.uniform(1.38, 1.58))
        y = amplitude * np.maximum(1 - 1 / (refractive_index**2 * x**2), 0)
        old = np.zeros_like(x)
        parameters = {"amplitude": amplitude, "refractive_index": refractive_index,
                      "threshold_beta": 1 / refractive_index}
    elif generator == "casimir_pressure":
        y = -amplitude / x**4
        old = np.zeros_like(x)
        parameters = {"amplitude": amplitude}
    elif generator == "parity_asymmetry":
        asymmetry = float(rng.uniform(-0.72, -0.38))
        y = amplitude * (1 + asymmetry * np.cos(x))
        old = np.full_like(x, amplitude)
        parameters = {"amplitude": amplitude, "asymmetry": asymmetry}
    elif generator == "anderson_envelope":
        length = float(rng.uniform(0.65, 1.35))
        y = amplitude * np.exp(-2 * np.abs(x) / length)
        old = np.full_like(x, float(np.mean(y)))
        parameters = {"amplitude": amplitude, "localization_length": length}
    elif generator == "pound_rebka":
        slope = float(rng.uniform(0.85, 1.15))
        y = amplitude * slope * x
        old = np.zeros_like(x)
        parameters = {"amplitude": amplitude, "rescaled_slope": slope}
    elif generator == "fano_profile":
        q = float(rng.uniform(0.65, 1.85)); shift = float(rng.uniform(-0.25, 0.25))
        width = float(rng.uniform(0.75, 1.25)); epsilon = (x - shift) / width
        y = amplitude * (q + epsilon) ** 2 / (1 + epsilon**2)
        old = amplitude * (1 + q*q) / (1 + epsilon**2)
        parameters = {"amplitude": amplitude, "q": q, "shift": shift, "width": width}
    elif generator == "aspect_correlation":
        visibility = float(rng.uniform(0.92, 0.99)); phase = float(rng.uniform(-0.05, 0.05))
        y = -visibility * np.cos(2 * x + phase)
        old = -(1 / np.sqrt(2)) * np.cos(2 * x + phase)
        parameters = {"visibility": visibility, "phase": phase,
                      "local_bound_visibility": float(1 / np.sqrt(2))}
    elif generator == "accelerating_universe":
        new = luminosity_distance_flat(x, 0.3, 0.7)
        incumbent = luminosity_distance_flat(x, 1.0, 0.0)
        y = amplitude * 5 * np.log10(new / incumbent)
        old = np.zeros_like(x)
        parameters = {"amplitude": amplitude, "omega_m": 0.3, "omega_lambda": 0.7}
    elif generator == "gw_chirp":
        y = amplitude * x ** (-3 / 8)
        old = np.full_like(x, float(np.median(y[x > np.median(x)])))
        parameters = {"amplitude": amplitude, "exponent": -3 / 8}
    else:
        raise KeyError(generator)
    return np.asarray(y, float), np.asarray(old, float), parameters


def main() -> None:
    assert len(CASES) == len({row["family_id"] for row in CASES})
    assert all(row["first_valid_year"] > 1900 for row in CASES)
    assert all(row["source_url"].startswith("https://") for row in CASES)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    clouds, clean, incumbent = [], [], []
    family_ids, record_ids, records = [], [], []
    for family_index, row in enumerate(CASES):
        for repeat in range(REPEATS):
            record_seed = SEED + 1000 * family_index + repeat
            rng = np.random.default_rng(record_seed)
            x = sample_x(row, rng)
            y_clean, y_old, parameters = evaluate(row["generator"], x, rng)
            assert np.isfinite(y_clean).all() and np.isfinite(y_old).all()
            noise_fraction = float(rng.uniform(0.005, 0.020))
            scale = max(float(np.std(y_clean)), 1e-10)
            hetero = 0.75 + 0.50 * (x - x.min()) / max(float(np.ptp(x)), 1e-12)
            y = y_clean + rng.normal(0, noise_fraction * scale, POINTS) * hetero
            record_id = f"{row['family_id']}__r{repeat:02d}"
            clouds.append(np.stack([x, y], axis=1).astype(np.float32))
            clean.append(y_clean.astype(np.float32)); incumbent.append(y_old.astype(np.float32))
            family_ids.append(row["family_id"]); record_ids.append(record_id)
            records.append({
                "record_id": record_id, "family_id": row["family_id"], "repeat": repeat,
                "seed": record_seed, "parameters": parameters,
                "noise_fraction": noise_fraction, "n": POINTS,
            })

    registry_path = DATA / "case_study_expansion_registry.json"
    registry_path.write_text(json.dumps({
        "status": "post-score source-screened development expansion",
        "selection_timing": "selected after the original 55-family benchmark was scored",
        "claim_boundary": "formula/simulator routing audit; not an enlarged primary AUROC and not real-measurement discovery",
        "cases": CASES, "theory_only_not_encoded_as_pointclouds": THEORY_ONLY,
    }, indent=2) + "\n", encoding="utf-8")
    archive_path = DATA / "case_study_expansion_pointclouds.npz"
    np.savez_compressed(
        archive_path, X=np.asarray(clouds, np.float32), clean_y=np.asarray(clean, np.float32),
        incumbent_y=np.asarray(incumbent, np.float32), family_ids=np.asarray(family_ids),
        record_ids=np.asarray(record_ids),
    )
    records_path = DATA / "case_study_expansion_records.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    output = {
        "status": "PASS", "seed": SEED, "families": len(CASES),
        "records": len(clouds), "records_per_family": REPEATS, "points_per_record": POINTS,
        "theory_only_cases": len(THEORY_ONLY),
        "files": {
            registry_path.name: {"sha256": sha256(registry_path), "bytes": registry_path.stat().st_size},
            archive_path.name: {"sha256": sha256(archive_path), "bytes": archive_path.stat().st_size},
            records_path.name: {"sha256": sha256(records_path), "bytes": records_path.stat().st_size},
        },
        "original_benchmark_mutated": False,
        "claim_boundary": "post-score development cases; descriptive component scores only",
    }
    manifest_path = RESULTS / "case_study_expansion_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
