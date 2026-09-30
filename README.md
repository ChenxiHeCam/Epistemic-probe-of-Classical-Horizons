# EPOCH: Epistemic Probe of Classical Horizons

This repository accompanies **"Recognizing the predictive boundary of
historically restricted physical knowledge"** (Chenxi He and Jingqiu Chen).
EPOCH tests whether a body of physical knowledge bounded by a date still
accounts for a set of measurements, where that compatibility ends, and whether
the theory that historically succeeded it removes the discrepancy.

Terms used throughout this repository:

- The **knowledge horizon** is a calendar date; only physical relations
  recorded before it are admitted to training and to the laws a test may fit.
  Here it is 1899.
- The **incumbent** is the pre-horizon law that a test examines for a given
  dataset; the incumbent prediction is its numerical prediction for the
  measured quantity.
- A **point cloud** is the set of `(x, y)` pairs of one relation, normalized to
  remove scale and stripped of formula, units, labels and ordering.
- A **law family** is the set of point clouds derived from one physical law
  traced to one dated source; a **mechanism group** is the set of measured
  series produced by one physical mechanism. AUROCs are computed, and bootstrap
  intervals resampled, over families or groups.
- **Frozen** means fixed, and recorded with a time stamp and file hash, before
  any test data were scored.

EPOCH reports three kinds of evidence separately, never as a single averaged
score:

1. the **learned component**, a neural encoder trained only on point clouds
   from pre-horizon relations, with no example of a departure, which scores a
   new point cloud by its distance from them;
2. the **four-form component**, which fits the best of power, exponential,
   linear and constant forms to the interior (the stretch of a dataset over
   which an admitted pre-horizon form fits well) and measures how far the
   frontier (the remainder) departs from the extrapolation; and
3. the **theory-conditioned component**, which computes the residuals of the
   measurements from the incumbent prediction, scaled by the declared
   measurement errors, and compares their size with a null distribution
   simulated from that prediction on the same grid with the same errors.

A fixed routing rule decides before scoring which component evaluates a
dataset, and each test returns one of four verdicts: compatible over the
measured range; localized incompatibility, with an interval for the boundary;
whole-range incompatibility, with no identifiable boundary; or abstention.

The learned component receives only a point cloud at query time. The pre-1900
formula graph is a training-only teacher; post-1900 formulas, dates, law
names, units and anomaly templates are absent at inference.

## Principal results

Native controls are the nine pre-1900 groups as originally generated;
generator-matched controls are the same groups regenerated with the sampling
and noise process of the later families; the generator-gated set draws later
formulas from the internal archive by the same generator as the pre-1900
corpus; metrological groups are calibration artefacts or instrument transfer
curves. The 38 and 55 later families are the curated, citation-anchored set.
A breakdown family is a set of synthetic curves built by imposing one kind of
departure, such as saturation, a step or a kink, on a classical form; such
curves are used only for evaluation.

- The large encoder, a 20,908,547-parameter dual encoder, was trained on
  880,896 point clouds from 3,441 source-audited pre-1900 formula reductions
  (each formula rewritten in dimensionless one-dimensional form). Its
  query-time point student has 9,868,672 parameters.
- Large encoder, native controls, 38 families from 1901–1950 vs nine pre-1900
  groups: AUROC 0.921 [0.833, 0.994]; before optimization 0.589.
- Large encoder, native controls, 55 families from 1901–1996 vs nine pre-1900
  groups: AUROC 0.925 [0.851, 0.982]; before optimization 0.649.
- Large encoder, generator-matched controls (registered after the primary
  scores were opened), 55 later vs nine pre-1900 groups: AUROC 0.857
  [0.741, 0.945]; before optimization 0.545.
- Large encoder, native generator-gated set of 1901–1950 formulas vs nine
  pre-1900 groups: AUROC 0.536 [0.424, 0.643]; before optimization 0.594
  [0.436, 0.756].
- Four-form component, frozen in advance, 55 later vs 11 pre-1900 families:
  AUROC 0.888 [0.800, 0.959] with no learned representation; in the same test
  the compact encoder (a width-128 set transformer trained on 205 pre-1900
  point clouds from 40 families) gives 0.669 [0.534, 0.798].
- Compact encoder trained on pre-1900 laws vs the generic-function control
  encoder, 12 breakdown and 12 classical families: family AUROC 0.903
  [0.729, 1.000] vs 0.375 [0.167, 0.618].
- Frozen evaluation at the 0.5 screening threshold, 200 formula records and 30
  synthetic breakdowns: AUROC 0.88.
- Four-form component at the calibrated threshold, 12 breakdown and 12
  classical families: true-positive rate 0.271 [0.073, 0.500]; 0/96 false
  positives at the 5%-targeted threshold.
- Advancing the horizon on identical data grids, that is rescoring the same
  measurements after replacing the incumbent prediction by the prediction of
  the theory that superseded it: the old horizon is rejected in all five
  cases, and 4/5 old-horizon discrepancies clear under the successful
  successor prediction; 5/5 wrong-shape and 5/5 wrong-mechanism predictions
  remain rejected.
- Frozen components on sixteen unmodified NIST measured series, 5 post-1900
  vs 9 classical or metrological groups: learned AUROC 0.62 [0.29, 0.89];
  four-form AUROC 0.78 [0.38, 1.00]; both intervals include 0.5. All 128 view
  scores are released without a threshold decision. The two highest learned
  scores belong to instrument records, a pattern consistent with the
  observation process setting the largest score differences.
- Boundary localization on 900 held-out simulations, with intervals
  calibrated separately at each sample size: 90% interval coverage 91.3%
  [89.3%, 93.0%].
- Simulated instrument effects at maximum severity: false-positive rate 0.40
  for saturation, 0.63 for censoring and at most 0.11 for all other tested
  effects.

Results are reduced at the law-family or mechanism-group level. The
machine-readable outputs, registrations and hashes used in the manuscript are
included under `results/` and `learned_v3/`; `results/evidence_manifest.json`
is the evidence ledger linking each reported number to its script and output.

## Repository layout

```text
learned_v3/      v3 architecture, training, evaluation and portable inference
models/          query-time point student of the large encoder (1899 horizon)
                 and its inference memory
engine/          statistical calibration, temporal audits, real-measurement
                 case construction and validators
casestudies/     historical and robustness experiments
data/            public inputs, generated evaluation point clouds and the
                 44-file NIST real-measurement inventory with hashes
results/         frozen numerical outputs and evidence manifests
figures/         publication figures and the plot_*.py scripts that draw them
figure_scripts/  matplotlib generators for the remaining manuscript figures
protocols/       confirmatory protocol template and the historical-case test
                 objects, with their JSON schema
paper/           LaTeX sources, bibliography and compiled PDFs of the
                 manuscript and Supplementary Information
```

Files retained from the original 2026-08 release support the earlier detector.
The learned results reported in the paper come from the `epoch_1899_*` assets
and the v3 registrations.

## Portable learned inference

Install the runtime dependencies and score the included benchmark of 55
post-1900 law families:

```bash
python -m pip install -r requirements.txt
python learned_v3/inference.py \
  --input data/post1900_formula_pointclouds.npz \
  --output output/portable_post1900_scores.json
```

For a single dataset, pass a two-column CSV or whitespace-delimited text file.
The script applies the frozen per-cloud normalization and deterministic
64-point sampling, then returns the four calibrated distances (each a rank on
its pre-1900 calibration distribution) and their unweighted mean,
`one_class_ensemble`, which is the learned score.

## Real-measurement series

The sixteen NIST series are frozen from the downloaded files in
`data/real_measurement_benchmark/raw/` and scored with both components frozen:

```bash
python engine/build_real_measurement_cases.py      # -> data/real_measurement_case_*.{npz,json}
python engine/evaluate_real_measurement_cases.py   # -> results/real_measurement_case_evaluation.json
python figures/plot_real_measurement_cases.py      # -> figures/figure_real_measurement_cases.*
```

Column semantics, the 1899 incumbent description and the mechanism group of
every series are declared in `engine/build_real_measurement_cases.py` before
any score is computed. `results/real_measurement_case_view_scores.csv` holds
the raw per-view learned and four-form scores; a view is one resampled or
rescaled copy of a series.

## Manuscript

`paper/manuscript.tex` and `paper/supplementary.tex` are the sources of
record; compiled PDFs sit beside them. Rebuild with

```bash
python paper/build.py
```

which needs `pdflatex` and `bibtex`. Figures are read from `figures/`.

## Release verification

Verify the byte identity of every tracked file against the deposited release:

```bash
python verify_release.py
```

`build_release_manifest.py` regenerates `release_manifest.json` from the
Git-tracked files when a new release is cut.

## Model and data records

The query-time point student and its pre-1900 inference memory are stored in
`models/`, together with SHA-256 sidecars. The full training checkpoint and
large point-cloud/AST archives are distributed through the versioned archive:

- GitHub: https://github.com/ChenxiHeCam/Epistemic-probe-of-Classical-Horizons
- Zenodo, this release: https://doi.org/10.5281/zenodo.22742368
- Zenodo, concept DOI (latest version): https://doi.org/10.5281/zenodo.22138620

## License

MIT. See `LICENSE`.
