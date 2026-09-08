"""Sensitivity of the copper clock-advance result to the reference-curve error floor."""

import json
from pathlib import Path

import numpy as np

from theory_clock_experiment import ROOT, ClockCase, bootstrap_gof, cp_reference, fit_cp_new, fit_cp_old


def main():
    x, y = cp_reference()
    rows = []
    for relative_floor in [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
        sigma_log = np.full(len(x), np.log1p(relative_floor))
        case = ClockCase("specific_heat_Cu", x, y, sigma_log, fit_cp_old, fit_cp_new, "NIST reference polynomial")
        old, old_stat, old_p, _ = bootstrap_gof(case, fit_cp_old, 1000, np.random.default_rng(1000), True, False)
        new, new_stat, new_p, _ = bootstrap_gof(case, fit_cp_new, 1000, np.random.default_rng(2000), True, False)
        rows.append({
            "relative_error_floor": relative_floor,
            "old_statistic": old_stat,
            "old_pvalue": old_p,
            "advanced_statistic": new_stat,
            "advanced_pvalue": new_p,
            "score_collapse_ratio": old_stat / new_stat,
            "paired_success": old_p <= 0.05 and new_p > 0.05,
        })
        print(relative_floor, old_p, new_p, old_stat / new_stat)
    output = ROOT / "results" / "specific_heat_error_sensitivity.json"
    output.write_text(json.dumps({
        "status": "sensitivity analysis; NIST reference-polynomial covariance is not available locally",
        "interpretation": "A conclusion that the advanced horizon clears the data must be stable over a defensible, externally justified error floor.",
        "rows": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
