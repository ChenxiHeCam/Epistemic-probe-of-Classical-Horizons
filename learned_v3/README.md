# Positive-only learned component v3

The frozen model aligns a six-block Set Transformer point-cloud student with a
training-only formula-graph teacher. All optimization examples belong to the
pre-1900 admitted horizon; no artificial breakdown, saturation, step, kink or
crossover class is supplied.

The complete architecture has 20,908,547 parameters. Query-time inference uses
the 9,868,672-parameter point student, four frozen one-class distances and the
pre-1900 calibration distributions packaged in `models/`.

Run the included 55-family temporal benchmark:

```bash
python learned_v3/inference.py \
  --input data/post1900_formula_pointclouds.npz \
  --output output/portable_post1900_scores.json
```

The output gives continuous record and family scores. `one_class_ensemble` is
the unweighted mean of four pre-1900-calibrated ranks; it is not a probability
that a dataset contains new physics.

The compact checkpoint and memory reproduce all 55 deposited family scores
within the fixed cross-device tolerance:

```bash
python learned_v3/verify_portable_inference.py
```

`EPOCH_LEARNED_V3_RUNBOOK.md` and the JSON registrations record the frozen data
views, architecture, optimization schedule, score construction and evaluation
order. The training and full-corpus evaluation programs are retained alongside
the portable query path.
