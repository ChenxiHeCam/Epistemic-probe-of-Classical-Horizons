# EPOCH: Epistemic Probe of Classical Horizons

This repository accompanies **“Auditing predictive boundaries from
historically restricted physical knowledge.”** EPOCH asks whether observations
remain predictable from a declared historical knowledge horizon, where that
predictive compatibility ends, and whether a successor theory removes the
discrepancy.

EPOCH combines three complementary components:

1. a positive-only learned representation of admitted physical relations;
2. data-anchored extrapolation from an observed incumbent regime; and
3. theory-conditioned residual tests against a declared physical prediction.

The learned query path receives only an unordered numeric point cloud. The
pre-1900 formula graph is a training-only teacher; post-1900 formulas, dates,
law names, units and anomaly templates are absent at inference.

## Current results

- The 20,908,547-parameter dual encoder was trained on 880,896 point clouds
  from 3,441 pre-1900 formula reductions. Its query-time point student has
  9,868,672 parameters.
- Against held-out pre-1900 groups, the frozen learned score ranks 38 physical
  law families from 1901–1950 with AUROC 0.921 and all 55 post-1900 families
  with AUROC 0.925. The same architecture before optimization reaches 0.649.
- A sampling- and noise-matched diagnostic gives AUROC 0.857; the independent
  frozen four-form component gives AUROC 0.888 on the 55-family audit.
- In the paired historical clock experiment, all five old-horizon predictions
  are rejected and four of five discrepancies become compatible after the
  historically successful successor prediction is admitted.
- Size-conditional boundary intervals attain 91.3% empirical coverage at 90%
  nominal coverage on held-out simulations.

Results are reduced at the source-law family or physical-mechanism level. The
machine-readable outputs, registrations and hashes used in the manuscript are
included under `results/` and `learned_v3/`.

## Repository layout

```text
learned_v3/      v3 architecture, training, evaluation and portable inference
models/          compact 1899-horizon student weights and inference memory
engine/          statistical calibration, temporal audits and validators
casestudies/     historical and robustness experiments
data/            public inputs and generated evaluation point clouds
results/         frozen numerical outputs and evidence manifests
figures/         publication figures and their generating scripts
paper/           current manuscript and Supplementary Information sources
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

Verify the byte identity of every file in the release:

```bash
python verify_release.py
```

## Model and data records

The compact query model and its pre-1900 inference memory are stored in
`models/`, together with SHA-256 sidecars. The full training checkpoint and
large point-cloud/AST archives are distributed through the versioned archive:

- GitHub: https://github.com/ChenxiHeCam/Epistemic-probe-of-Classical-Horizons
- Zenodo: https://doi.org/10.5281/zenodo.22138621

The archive DOI currently resolves to the first public release; the manuscript
release is deposited as a new version so that its Git commit, model hashes and
data manifests remain permanently linked.

## License

MIT. See `LICENSE`.
