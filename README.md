# EPOCH: Epistemic Probe of Classical Horizons

This repository accompanies **"Recognizing the predictive boundary of
historically restricted physical knowledge"** (Chenxi He and Jingqiu Chen).
EPOCH asks whether observations remain predictable from a declared historical
knowledge horizon, where that predictive compatibility ends, and whether a
successor theory removes the discrepancy.

EPOCH reports three kinds of evidence separately rather than averaging them:

1. a positive-only learned representation of admitted physical relations;
2. data-anchored extrapolation from an observed incumbent regime; and
3. theory-conditioned residual tests against a declared physical prediction.

The learned query path receives only an unordered numeric point cloud. The
pre-1900 formula graph is a training-only teacher; post-1900 formulas, dates,
law names, units and anomaly templates are absent at inference.

## Principal results

- The 20,908,547-parameter dual encoder was trained on 880,896 point clouds
  from 3,441 source-audited pre-1900 formula reductions. Its query-time point
  student has 9,868,672 parameters.
- Against held-out pre-1900 groups, the frozen learned score ranks 38 physical
  law families from 1901–1950 with AUROC 0.921 [0.833, 0.994] and all 55
  post-1900 families with AUROC 0.925 [0.851, 0.982]. The same architecture
  before optimization reaches 0.649.
- Regenerating the nine control groups under the later benchmark's sampling and
  noise process gives AUROC 0.857 [0.741, 0.945]; the independently frozen
  four-form extrapolation component gives AUROC 0.888 [0.800, 0.959] with no
  learned representation.
- In the paired historical clock experiment on identical data grids, all five
  old-horizon predictions are rejected and four of five discrepancies become
  compatible once the successful successor prediction is admitted, while
  wrong-shape and wrong-mechanism alternatives stay rejected.
- On sixteen unmodified NIST measurement series (five post-1900 mechanism
  groups versus nine classical or metrological groups) the frozen components
  give learned AUROC 0.62 [0.29, 0.89] and four-form AUROC 0.78 [0.38, 1.00].
  Raw per-view scores are released without any threshold decision; the leading
  axis of variation is the observation process rather than the period of the
  underlying physics.
- Size-conditional boundary intervals attain 91.3% empirical coverage at 90%
  nominal coverage on 900 held-out simulations.

Results are reduced at the source-law family or physical-mechanism level. The
machine-readable outputs, registrations and hashes used in the manuscript are
included under `results/` and `learned_v3/`; `results/evidence_manifest.json`
is the evidence ledger linking each reported number to its script and output.

## Repository layout

```text
learned_v3/      v3 architecture, training, evaluation and portable inference
models/          compact 1899-horizon student weights and inference memory
engine/          statistical calibration, temporal audits, real-measurement
                 case construction and validators
casestudies/     historical and robustness experiments
data/            public inputs, generated evaluation point clouds and the
                 44-file NIST real-measurement benchmark with hashes
results/         frozen numerical outputs and evidence manifests
figures/         publication figures and the plot_*.py scripts that draw them
figure_scripts/  matplotlib generators for the remaining manuscript figures
paper/           LaTeX sources, bibliography and compiled PDFs of the
                 manuscript and Supplementary Information
```

Files retained from the original 2026-08 release support the legacy detector;
the current headline learned result is identified by the `epoch_1899_*` assets
and the v3 registrations.

## Portable learned inference

Install the runtime dependencies and score the included temporal benchmark:

```bash
python -m pip install -r requirements.txt
python learned_v3/inference.py \
  --input data/post1900_formula_pointclouds.npz \
  --output output/portable_post1900_scores.json
```

For a single dataset, pass a two-column CSV or whitespace-delimited text file.
The script applies the frozen per-cloud normalization and deterministic
64-point sampling before returning all four calibrated coordinates and their
unweighted ensemble.

## Real-measurement series

The sixteen NIST series are frozen from the downloaded files in
`data/real_measurement_benchmark/raw/` and scored without any training:

```bash
python engine/build_real_measurement_cases.py      # -> data/real_measurement_case_*.{npz,json}
python engine/evaluate_real_measurement_cases.py   # -> results/real_measurement_case_evaluation.json
python figures/plot_real_measurement_cases.py      # -> figures/figure_real_measurement_cases.*
```

Column semantics, the 1899 incumbent description and the mechanism group of
every series are declared in `engine/build_real_measurement_cases.py` before
any score is computed. `results/real_measurement_case_view_scores.csv` holds
the raw per-view learned and four-form scores.

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

The compact query model and its pre-1900 inference memory are stored in
`models/`, together with SHA-256 sidecars. The full training checkpoint and
large point-cloud/AST archives are distributed through the versioned archive:

- GitHub: https://github.com/ChenxiHeCam/Epistemic-probe-of-Classical-Horizons
- Zenodo, this release: https://doi.org/10.5281/zenodo.22742368
- Zenodo, concept DOI (latest version): https://doi.org/10.5281/zenodo.22138620

## License

MIT. See `LICENSE`.
