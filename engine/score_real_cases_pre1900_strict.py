"""Descriptive real-case scores from the strict classical-only zero-unit encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from evaluate_pre1900_strict_encoder import DATA, ROOT, encode, knn_scores, load_encoder


CHECKPOINT = ROOT / "data" / "encoder_pre1900_strict_zero_units.pt"
OUTPUT = ROOT / "results" / "pre1900_strict_real_case_scores.json"


def load_two_columns(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value[0].isalpha():
            continue
        fields = value.replace(",", " ").split()
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except (ValueError, IndexError):
            pass
    return np.asarray(rows, np.float32)


def michelson():
    path = ROOT / "data" / "real_measurement_benchmark" / "raw" / "MICHELSO.DAT"
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            speed, _temperature, day, _am_pm, _set = map(float, fields)
        except ValueError:
            continue
        rows.append((day, speed))
    if len(rows) != 100:
        raise ValueError(f"expected 100 Michelson rows, found {len(rows)}")
    return np.asarray(rows, np.float32)


def main():
    archive = np.load(DATA, allow_pickle=True)
    split = np.asarray(archive["splits"]).astype(str)
    model, checkpoint = load_encoder(CHECKPOINT)
    if checkpoint["uses_dimension_metadata"]:
        raise ValueError("real-case script requires the zero-unit checkpoint")
    embeddings = encode(model, archive["X"], None)
    reference = embeddings[split == "train"]
    validation = embeddings[split == "validation"]
    validation_family_count = len(set(
        np.asarray(archive["family_ids"])[split == "validation"].astype(str)
    ))
    validation_knn = knn_scores(reference, validation)
    covariance = LedoitWolf().fit(reference)
    validation_mahalanobis = covariance.mahalanobis(validation)

    firas = load_two_columns(ROOT / "data" / "firas_monopole.txt")
    coefficients = [-1.91844, -0.15973, 8.61013, -18.996, 21.9661, -12.7328, 3.54322, -0.3797, 0]
    temperature = np.logspace(np.log10(4), np.log10(300), 120)
    heat_capacity = 10 ** np.clip(
        sum(value * np.log10(temperature) ** power for power, value in enumerate(coefficients)), -30, 30
    )
    boyle = np.asarray(json.loads((ROOT / "data" / "boyle_1662.json").read_text(encoding="utf-8")), np.float32)
    cases = {
        "FIRAS_blackbody": firas,
        "copper_specific_heat": np.stack([temperature, heat_capacity], axis=1).astype(np.float32),
        "Boyle_1662_control": boyle,
        "Michelson_1879_control": michelson(),
    }
    threshold_knn = float(np.quantile(validation_knn, 0.95, method="higher"))
    threshold_mahalanobis = float(np.quantile(validation_mahalanobis, 0.95, method="higher"))
    results = {}
    for name, cloud in cases.items():
        z = encode(model, cloud[None], None)
        knn = float(knn_scores(reference, z)[0])
        mahalanobis = float(covariance.mahalanobis(z)[0])
        results[name] = {
            "n": len(cloud),
            "knn_score": knn,
            "knn_conformal_p": float((1 + np.sum(validation_knn >= knn)) / (len(validation_knn) + 1)),
            "knn_alert_at_exploratory_validation_95pct": bool(knn > threshold_knn),
            "mahalanobis_score": mahalanobis,
            "mahalanobis_conformal_p": float(
                (1 + np.sum(validation_mahalanobis >= mahalanobis)) / (len(validation_mahalanobis) + 1)
            ),
            "mahalanobis_alert_at_exploratory_validation_95pct": bool(mahalanobis > threshold_mahalanobis),
        }
    output = {
        "status": "descriptive real-case scoring; not confirmatory",
        "checkpoint": CHECKPOINT.name,
        "reference": {"train_records": len(reference), "validation_records": len(validation),
                      "threshold_knn": threshold_knn, "threshold_mahalanobis": threshold_mahalanobis},
        "results": results,
        "limitations": [
            (
                f"Only {len(validation)} validation records from {validation_family_count} families; "
                f"the smallest attainable conformal p-value is 1/{len(validation) + 1}."
            ),
            "FIRAS and the specific-heat curve differ from the formula-derived training measurement process.",
            "Bertozzi and Onnes are omitted because they have fewer than 20 points.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
