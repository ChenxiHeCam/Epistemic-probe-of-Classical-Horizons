# Learned component v3

This directory holds the architecture, training, evaluation and portable
inference code for the learned component of EPOCH: a neural encoder trained
only on point clouds from relations recorded before the 1899 knowledge
horizon, with no example of a departure (positive-only training), which
scores a new point cloud by its distance from those relations. A point cloud
is the set of `(x, y)` pairs of one relation, normalized and stripped of
formula, units, labels and ordering.

The frozen model pairs a six-block Set Transformer point-cloud encoder, the
student and the only network used at query time, with a formula-graph
encoder, the teacher, used only during training. Frozen means fixed, and
recorded with a time stamp and file hash, before any test data were scored.
Every optimization example comes from the pre-1900 admitted corpus; no
artificial breakdown, saturation, step, kink or crossover class is supplied.

The complete architecture has 20,908,547 parameters. Query-time inference uses
the 9,868,672-parameter point student, four frozen distances to the pre-1900
reference memory and the pre-1900 calibration distributions packaged in
`models/`.

Run the included benchmark of 55 post-1900 law families (a law family is the
set of point clouds derived from one physical law traced to one dated source):

```bash
python learned_v3/inference.py \
  --input data/post1900_formula_pointclouds.npz \
  --output output/portable_post1900_scores.json
```

The output gives continuous scores per record and per family.
`one_class_ensemble` is the unweighted mean of the four distances after each
has been converted to a rank on its pre-1900 calibration distribution. Higher
values mean farther from the pre-1900 relations; the score orders datasets
against each other and against the pre-1900 calibration set.

The query-time point student and memory reproduce all 55 deposited family
scores within the fixed cross-device tolerance:

```bash
python learned_v3/verify_portable_inference.py
```

`EPOCH_LEARNED_V3_RUNBOOK.md` and the JSON registrations record the frozen
data views, architecture, optimization schedule, score construction and
evaluation order. The training and full-corpus evaluation programs are kept
beside the portable query path.
