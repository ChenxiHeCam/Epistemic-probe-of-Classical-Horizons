"""Freeze a real-measurement case set for the EPOCH components.

Every record is an unmodified NIST measurement series, hash-bound to the
downloaded file.  This module fixes, for each dataset, the column semantics,
the 1899 incumbent description and the mechanism group that acts as the
inferential unit.  It performs no scoring and no training.

Horizon classes are declared from the measured phenomenon and its primary
source, before any EPOCH score is computed.  CHWIRUT2 and CLARK2 are recorded
as subsets of CHWIRUT1 and CLARK3 respectively and share their mechanism group,
so neither adds an independent unit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "real_measurement_benchmark"
OUT_ARCHIVE = ROOT / "data" / "real_measurement_case_pointclouds.npz"
OUT_REGISTRY = ROOT / "data" / "real_measurement_case_registry.json"
MAX_POINTS = 2000

# x_col / y_col index into the parsed numeric block of the raw NIST file.
# x_col == -1 means the abscissa is reconstructed from separate date columns.
CASES = [
    dict(
        dataset="HAHN1", mechanism_group="cryogenic_thermal_expansion",
        x_col=0, y_col=1,
        x_name="temperature (K)", y_name="coefficient of thermal expansion",
        column_note="the numeric block stores (temperature, coefficient); the header text lists the "
                    "response first, so the order was fixed against the certified StRD values "
                    "(first point: 24.41 K, 0.591)",
        horizon_class="post_1900_mechanism",
        incumbent_1899="temperature-independent expansion coefficient from classical equipartition of lattice modes",
        why="the measured coefficient falls towards zero below the Debye temperature, which equipartition cannot produce",
        source="T. Hahn, copper thermal expansion study, NIST (1970s); NIST StRD HAHN1",
    ),
    dict(
        dataset="BENNETT5", mechanism_group="superconducting_flux_creep",
        x_col=1, y_col=0,
        x_name="log time (ln minutes)", y_name="magnetization M(t)",
        horizon_class="post_1900_mechanism",
        incumbent_1899="no persistent screening currents; classical Ohmic decay of an induced magnetization",
        why="magnetization relaxation of a superconductor has no pre-1900 account",
        source="L. Bennett, L. Swartzendruber and H. Brown, NIST (April 1994); NIST StRD BENNETT5",
    ),
    dict(
        dataset="BENNETT6", mechanism_group="superconducting_flux_creep",
        x_col=1, y_col=0,
        x_name="log time (ln minutes)", y_name="magnetization M(t)",
        horizon_class="post_1900_mechanism",
        incumbent_1899="no persistent screening currents; classical Ohmic decay of an induced magnetization",
        why="magnetization relaxation of a superconductor has no pre-1900 account",
        source="L. Bennett, L. Swartzendruber and H. Brown, NIST (April 1994); NIST StRD BENNETT6",
    ),
    dict(
        dataset="BENNETT7", mechanism_group="superconducting_flux_creep",
        x_col=0, y_col=1,
        x_name="log time (ln minutes)", y_name="magnetization M(t)",
        horizon_class="post_1900_mechanism",
        incumbent_1899="no persistent screening currents; classical Ohmic decay of an induced magnetization",
        why="magnetization relaxation of a superconductor has no pre-1900 account",
        source="L. Bennett, L. Swartzendruber and H. Brown, NIST (April 1994); NIST StRD BENNETT7",
    ),
    dict(
        dataset="CLARK3", mechanism_group="bose_gas_density",
        x_col=1, y_col=0,
        x_name="position", y_name="cold Bose-gas density",
        horizon_class="post_1900_mechanism",
        incumbent_1899="Maxwell-Boltzmann density profile of a classical gas",
        why="the density profile of a degenerate Bose gas requires quantum statistics",
        source="C. Clark, Physics Laboratory, NIST (1995); NIST Dataplot CLARK3, a 1/5 decimation of CLARK2",
    ),
    dict(
        dataset="ROSZMAN1", mechanism_group="rydberg_series",
        x_col=0, y_col=1,
        x_name="principal quantum number", y_name="term energy (cm-1)",
        horizon_class="classical_or_metrology",
        incumbent_1899="empirical Rydberg series with a fitted series constant (Rydberg 1888)",
        why="the file is a bare (n, term energy) block for a sulfur Rydberg series; "
            "the measured curve follows an inverse-square series that was available "
            "empirically before 1900, so the declared horizon already predicts it",
        column_note="the file carries no header; the certified StRD response (the quantum defect "
                    "itself) is absent, and the second column is the StRD predictor",
        source="R. Roszman, quantum defects for sulfur, NIST; NIST StRD ROSZMAN1 predictor block",
    ),
    dict(
        dataset="KIM", mechanism_group="semiconductor_magnetotransport",
        x_col=0, y_col=1,
        x_name="magnetic field strength", y_name="reduced conductivity (type 1)",
        horizon_class="post_1900_mechanism",
        incumbent_1899="single-carrier Drude magnetoconductivity",
        why="the two-carrier reduction of the measured curve rests on band theory",
        source="Kim, electron mobility analysis, NIST (June 1992); NIST Dataplot KIM",
    ),
    dict(
        dataset="THURBER", mechanism_group="semiconductor_mobility",
        x_col=1, y_col=0,
        x_name="log carrier density", y_name="electron mobility",
        horizon_class="post_1900_mechanism",
        incumbent_1899="classical free-electron transport with density-independent mobility",
        why="the measured mobility-density relation rests on band structure and impurity scattering",
        source="R. Thurber, semiconductor electron mobility, NIST; NIST StRD THURBER",
    ),
    dict(
        dataset="CHWIRUT1", mechanism_group="ultrasonic_attenuation",
        x_col=1, y_col=0,
        x_name="metal distance", y_name="ultrasonic response",
        horizon_class="classical_or_metrology",
        incumbent_1899="attenuated elastic-wave propagation with geometric spreading",
        why="classical acoustics covers the measured attenuation",
        source="D. Chwirut, ultrasonic reference-block calibration, NIST; NIST StRD CHWIRUT1",
    ),
    dict(
        dataset="ECKERLE4", mechanism_group="circular_interference",
        x_col=0, y_col=1,
        x_name="wavelength (nm)", y_name="transmittance",
        horizon_class="classical_or_metrology",
        incumbent_1899="classical wave-optical interference in a circular aperture",
        why="a sharp transmission resonance is morphologically unusual but fully classical; this is the shape-versus-physics hard control",
        source="K. Eckerle, circular interference transmittance, NIST; NIST StRD ECKERLE4",
    ),
    dict(
        dataset="PONTIUS", mechanism_group="load_cell_elasticity",
        x_col=1, y_col=0,
        x_name="applied load", y_name="deflection",
        horizon_class="classical_or_metrology",
        incumbent_1899="Hookean elastic response with a small quadratic correction",
        why="classical elasticity covers the measured deflection",
        source="P. Pontius, NIST load-cell calibration; NIST StRD PONTIUS",
    ),
    dict(
        dataset="KIRBY2", mechanism_group="linewidth_standard",
        x_col=1, y_col=0,
        x_name="nominal line width", y_name="measured line width",
        horizon_class="classical_or_metrology",
        incumbent_1899="smooth monotone instrument response",
        why="an electron-microscope line-width standard is a metrology transfer curve, not a physical law",
        source="R. Kirby, electron-microscope line-width standards, NIST; NIST StRD KIRBY2",
    ),
    dict(
        dataset="BERGER1", mechanism_group="radiographic_defect_calibration",
        x_col=1, y_col=0,
        x_name="in-lab defect size", y_name="in-field defect size",
        horizon_class="classical_or_metrology",
        incumbent_1899="smooth monotone instrument response",
        why="a radiographic calibration curve is a metrology transfer curve, not a physical law",
        source="Berger, Alaska pipeline ultrasonic defect calibration, NIST; NIST StRD BERGER1",
    ),
    dict(
        dataset="CALIBRATIONLINE", mechanism_group="optical_linewidth_calibration",
        x_col=0, y_col=1,
        x_name="line-width artifact", y_name="measured line width",
        horizon_class="classical_or_metrology",
        incumbent_1899="smooth monotone instrument response",
        why="an optical imaging calibration line is a metrology transfer curve, not a physical law",
        source="C. Croarkin, optical imaging linewidth calibration, NIST; NIST Dataplot CALIBRATIONLINE",
    ),
    dict(
        dataset="WATTODR3", mechanism_group="polymer_impedance",
        x_col=0, y_col=1,
        x_name="frequency", y_name="real part of complex impedance",
        horizon_class="classical_or_metrology",
        incumbent_1899="classical dielectric relaxation of a polymer",
        why="complex impedance dispersion is classical electromagnetism",
        source="Bates and Watts (1988) p. 280 polymer impedance; NIST Dataplot WATTODR3",
    ),
    dict(
        dataset="DZIUBA1", mechanism_group="standard_resistor_drift",
        x_col=-1, y_col=3,
        x_name="decimal date", y_name="standard resistor value",
        horizon_class="classical_or_metrology",
        incumbent_1899="slow monotone instrument drift about a stable reference value",
        why="a five-year standard-resistor record is an instrument stability series",
        source="R. Dziuba, standard resistors 1980-1985, NIST (July 1986); NIST Dataplot DZIUBA1",
    ),
]

SUBSETS = {
    "CHWIRUT2": "subset of CHWIRUT1; shares mechanism group ultrasonic_attenuation and adds no independent unit",
    "CLARK2": "CLARK3 is its 1/5 decimation; shares mechanism group bose_gas_density and adds no independent unit",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_block(record: dict) -> np.ndarray:
    path = BENCH / record["file"]
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    for line in lines[record["start_line"] - 1: record["end_line"]]:
        parts = line.split()
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    width = max(len(row) for row in rows)
    return np.asarray([row for row in rows if len(row) == width], float)


def main() -> None:
    inventory = json.loads((BENCH / "inventory_report.json").read_text(encoding="utf-8"))
    meta = {row["id"]: row for row in inventory["datasets"]}

    clouds, ids, groups, registry = [], [], [], []
    for case in CASES:
        record = meta[case["dataset"]]
        block = numeric_block(record)
        if case["x_col"] == -1:
            month, day, year = block[:, 0], block[:, 1], block[:, 2]
            x = (1900.0 + year) + (month - 1.0) / 12.0 + (day - 1.0) / 372.0
        else:
            x = block[:, case["x_col"]]
        y = block[:, case["y_col"]]
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        available = int(len(x))
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        if available > MAX_POINTS:
            index = np.linspace(0, available - 1, MAX_POINTS).round().astype(int)
            x, y = x[index], y[index]
        cloud = np.column_stack([x, y]).astype(np.float32)
        clouds.append(cloud)
        ids.append(case["dataset"])
        groups.append(case["mechanism_group"])
        registry.append({
            **case,
            "points_used": int(len(cloud)),
            "points_available": available,
            "rows_in_file": int(record["rows"]),
            "file": record["file"],
            "file_sha256": record["sha256"],
            "x_range": [float(x.min()), float(x.max())],
            "y_range": [float(y.min()), float(y.max())],
        })

    lengths = [len(cloud) for cloud in clouds]
    payload = {f"cloud_{index}": cloud for index, cloud in enumerate(clouds)}
    np.savez_compressed(
        OUT_ARCHIVE,
        dataset_ids=np.asarray(ids),
        mechanism_groups=np.asarray(groups),
        lengths=np.asarray(lengths),
        **payload,
    )
    positive = sorted({r["mechanism_group"] for r in registry if r["horizon_class"] == "post_1900_mechanism"})
    negative = sorted({r["mechanism_group"] for r in registry if r["horizon_class"] == "classical_or_metrology"})
    OUT_REGISTRY.write_text(json.dumps({
        "status": "frozen real-measurement case registry; column semantics and horizon class fixed before scoring",
        "label_basis": "horizon class declared from the measured phenomenon and its primary source, never from an EPOCH score",
        "datasets": len(registry),
        "mechanism_groups": sorted(set(groups)),
        "post_1900_mechanism_groups": positive,
        "classical_or_metrology_groups": negative,
        "excluded_subset_files": SUBSETS,
        "max_points_per_record": MAX_POINTS,
        "cases": registry,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(registry)} datasets across {len(set(groups))} mechanism groups")
    print(f"  post-1900 groups: {len(positive)}   classical/metrology groups: {len(negative)}")
    print(f"archive sha256 {sha256(OUT_ARCHIVE)}")
    for row in registry:
        print(f"  {row['dataset']:16s} {row['horizon_class']:22s} "
              f"{row['mechanism_group']:34s} n={row['points_used']}")


if __name__ == "__main__":
    main()
