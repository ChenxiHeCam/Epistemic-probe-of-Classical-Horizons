"""Case-matched, dimensionally valid but mechanism-irrelevant horizon controls.

The earlier wrong-shape controls only ask whether a small generic function can
erase each mismatch.  Here every distractor is an actual physical functional
form with dimension-carrying nuisance parameters, but it is deliberately
assigned to the wrong mechanism.  This is a retrospective specificity stress
test, not a preregistered comparison of historically admissible theories.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import curve_fit, least_squares

from theory_clock_experiment import (
    ROOT,
    HorizonFit,
    bh_qvalues,
    bootstrap_gof,
    cases,
)


H = 6.62607015e-34
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
C_LIGHT = 299792458.0
PROTON_REST_MEV = 938.27208816


def firas_fermionic_spectrum(x, y, sigma):
    """Fermi-Dirac rather than bosonic occupation, with matched two-parameter scale."""

    def model(xv, log_amplitude, inverse_temperature_scale):
        exponent = np.clip(inverse_temperature_scale * xv, 1e-8, 60)
        return np.exp(log_amplitude) * xv**3 / (np.exp(exponent) + 1)

    params, _ = curve_fit(
        model,
        x,
        y,
        p0=[np.log(20), 0.5],
        sigma=sigma,
        absolute_sigma=True,
        bounds=([-20, 1e-4], [20, 10]),
        maxfev=20000,
    )
    prediction = model(x, *params)
    return HorizonFit(
        "fermionic occupation spectrum",
        prediction,
        lambda y_new: firas_fermionic_spectrum(x, y_new, sigma),
        {
            "log_amplitude": float(params[0]),
            "inverse_temperature_scale": float(params[1]),
        },
        "Same radiance and wavenumber dimensions as the Planck fit, but uses a Fermi-Dirac +1 denominator for photons.",
    )


def bertozzi_proton_mass_relativity(x, y, sigma):
    """Correct relativistic form with the wrong particle rest-energy scale."""

    prediction = 1 - 1 / (1 + x / PROTON_REST_MEV) ** 2
    return HorizonFit(
        "relativistic relation with proton rest energy",
        prediction,
        lambda y_new: bertozzi_proton_mass_relativity(x, y_new, sigma),
        {"fixed_rest_energy_MeV": PROTON_REST_MEV},
        "Input and rest energy are both MeV and beta squared is dimensionless; the proton scale is physically inapplicable to the electron beam.",
    )


def _einstein_heat_capacity(z):
    z = np.clip(np.asarray(z, float), 1e-5, 80)
    decay = np.exp(-z)
    return z**2 * decay / np.maximum((1 - decay) ** 2, 1e-300)


def copper_einstein_solid(x, y, sigma):
    """Einstein oscillator solid plus an electronic term, versus Debye acoustics."""

    log_y = np.log(np.clip(y, 1e-30, None))

    def prediction(params):
        plateau, theta_e, gamma = np.exp(params)
        values = plateau * _einstein_heat_capacity(theta_e / x) + gamma * x
        return np.clip(values, 1e-30, None)

    result = least_squares(
        lambda params: (log_y - np.log(prediction(params))) / sigma,
        x0=np.log([390, 250, 1e-3]),
        bounds=(np.log([10, 10, 1e-8]), np.log([5000, 5000, 10])),
        max_nfev=3000,
    )
    plateau, theta_e, gamma = np.exp(result.x)
    return HorizonFit(
        "Einstein oscillator plus electronic term",
        prediction(result.x),
        lambda y_new: copper_einstein_solid(x, y_new, sigma),
        {
            "plateau_J_per_kg_K": float(plateau),
            "einstein_temperature_K": float(theta_e),
            "gamma_J_per_kg_K2": float(gamma),
        },
        "All terms have heat-capacity units; an Einstein oscillator is a physical quantum solid model but lacks the Debye acoustic-mode continuum.",
    )


def onnes_arrhenius_transport(x, y, sigma):
    """Continuous thermally activated resistance rather than a phase transition."""

    def prediction(params):
        log_r0, activation_temperature = params
        return np.exp(np.clip(log_r0 - activation_temperature / x, -60, 20))

    result = least_squares(
        lambda params: (y - prediction(params)) / sigma,
        x0=[np.log(1.0), 20.0],
        bounds=([-30, 0], [30, 1e5]),
        max_nfev=10000,
    )
    return HorizonFit(
        "Arrhenius activated resistance",
        prediction(result.x),
        lambda y_new: onnes_arrhenius_transport(x, y_new, sigma),
        {
            "log_R0_ohm": float(result.x[0]),
            "activation_temperature_K": float(result.x[1]),
        },
        "R0 carries ohms and the activation scale carries kelvin; the response is continuous and cannot encode a superconducting transition.",
    )


def millikan_compton_recoil(x, y, sigma):
    """Frequency-dependent Compton recoil scale with a fitted voltage offset."""

    frequency_hz = x * 1e14
    recoil_voltage = (H * frequency_hz) ** 2 / (2 * M_E * C_LIGHT**2 * E_CHARGE)
    weight = 1 / sigma**2
    offset = np.sum(weight * (y - recoil_voltage)) / np.sum(weight)
    prediction = offset + recoil_voltage
    return HorizonFit(
        "Compton-recoil energy scale",
        prediction,
        lambda y_new: millikan_compton_recoil(x, y_new, sigma),
        {"offset_V": float(offset), "physical_constants": "2019 SI exact h, e; CODATA electron mass"},
        "The recoil energy divided by charge has volts and the offset has volts; this post-1900 mechanism is irrelevant to the photoelectric stopping-potential slope.",
    )


DISTRACTORS = {
    "FIRAS": firas_fermionic_spectrum,
    "Bertozzi": bertozzi_proton_mass_relativity,
    "specific_heat_Cu": copper_einstein_solid,
    "Onnes": onnes_arrhenius_transport,
    "Millikan": millikan_compton_recoil,
}

DIMENSION_AUDIT = {
    "FIRAS": "x and inverse-temperature scale form a dimensionless exponent; amplitude supplies radiance/x^3",
    "Bertozzi": "kinetic energy/rest energy is dimensionless; output beta^2 is dimensionless",
    "specific_heat_Cu": "theta/T is dimensionless; plateau and gamma*T both have heat-capacity units",
    "Onnes": "activation temperature/T is dimensionless; R0 supplies resistance units",
    "Millikan": "(h nu)^2/(m_e c^2 e) is a voltage; fitted offset is a voltage",
}


def main() -> None:
    rows = []
    for index, case in enumerate(cases()):
        log_space = case.name == "specific_heat_Cu"
        censored = case.name == "Onnes"
        fit, statistic, pvalue, valid = bootstrap_gof(
            case,
            DISTRACTORS[case.name],
            1000,
            np.random.default_rng(20261920 + index),
            log_space,
            censored,
        )
        rows.append(
            {
                "case": case.name,
                "distractor": fit.name,
                "statistic": statistic,
                "pvalue": pvalue,
                "bootstrap_valid": valid,
                "parameters": fit.fitted_parameters,
                "dimension_audit": DIMENSION_AUDIT[case.name],
                "assumptions": fit.assumptions,
                "remains_rejected_5pct": bool(pvalue <= 0.05),
            }
        )
        print(case.name, fit.name, pvalue, flush=True)

    qvalues = bh_qvalues([row["pvalue"] for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["bh_qvalue"] = float(qvalue)
        row["remains_rejected_fdr5"] = bool(qvalue <= 0.05)

    output = {
        "status": "retrospective physically dimensioned distractor control; not a historical-theory benchmark",
        "question": "Does admitting a dimensionally valid but mechanism-irrelevant physical form erase the old-horizon mismatch?",
        "selection_warning": "Distractors were designed after the canonical cases were known; p-values diagnose fit under the stated null and are not prospective model-selection evidence.",
        "cases": rows,
    }
    path = ROOT / "results" / "physical_irrelevant_horizon_controls.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
