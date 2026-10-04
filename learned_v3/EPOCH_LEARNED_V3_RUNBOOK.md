# EPOCH learned component v3: frozen two-horizon runbook

## Scientific question

Does a point-cloud representation optimized only against formula structure
admitted at a historical knowledge horizon assign larger one-class anomaly
scores to later formula families?  Does that anomaly score fall for the same
1901--1950 families after the knowledge horizon is advanced to 1950?

This is a temporal formula-shape transfer experiment.  It is not a real-data
discovery benchmark, a test of publication date, causal attribution to new
physics, or recovery of a successor law.

## Frozen data

- 19,605 finite formula reductions: 18,104 pre-1900 and 1,501 generator-gated
  1901--1950 candidates.
- 1,874,880 point clouds in 473 mmap shards, with 64, 128 or 256 native points.
- Phase A strict training: 880,896 pre-1900 clouds; calibration 11,648;
  internal test 62,336; temporal 1901--1950 test 96,064.
- Phase B strict joint training: the same 880,896 pre-1900 clouds plus 78,848
  pre-1950 increment clouds; calibration 20,160; internal test 71,040.
- Formula structure: 19,605 AST graphs, 275,722 nodes and 512,234 directed
  edges, joined by immutable `formula_index`.
- Curated temporal formula tests: 304 records from 38 families dated
  1901--1950 and 136 records from 17 post-1950 families.
- Raw symbolic pools contain 848,697 pre-1900 and 726,702 nominal pre-1950
  augmentation tasks.  These rows are not independent scientific units; the
  headline inferential unit is a source-law family or frozen relation group.
  They are staged for the separate symbolic/abduction objective and are not
  counted as optimizer examples for this point-cloud detector.  Here the
  verified chains supply formula structure, provenance and grouping, while the
  optimizer consumes the point-cloud/AST pairs listed above.

The pre-1900 strict subset is registry-backed.  The 1901--1950 increment is
time-separated and generator-gated, but does not yet have a complete dated
source registry.  Deformation, saturation, step, kink and crossover examples
are disabled in the headline learned runs.

## Architecture

The 20,908,547-parameter dual encoder contains:

1. A permutation-invariant six-block Set Transformer point student
   (`d_model=384`, 256-dimensional output, eight heads, four pooling tokens).
2. A training-only formula-graph teacher with four edge-aware local updates
   and two graph-local transformer layers.
3. Exactly one RBF KAN-style bottleneck (`96 x 8` basis grid) in a local graph
   update.  The remaining feed-forward modules use ordinary MLP/SwiGLU
   blocks.  KAN therefore augments one relation-update MLP; it does not replace
   the GNN.
4. A multi-positive cross-modal objective plus point-view invariance and
   variance regularization.  No anomaly class or deformation label is used.

At evaluation, the query path receives only a normalized unordered `(x, y)`
point cloud.  It receives no formula string, AST, units, date, law name or
post-cutoff label.

## Frozen one-class score

Four components are fit using admitted-horizon training and calibration data:

- top-8 cosine distance to known formula-graph embeddings;
- top-8 cosine distance to known identity centroids;
- top-8 cosine distance to empirical point-formula prototypes;
- deterministic shrinkage Mahalanobis distance to the point prototypes.

Each component is transformed by its pre-cutoff calibration-formula empirical
CDF.  Their unweighted arithmetic mean is the primary score.  Scores are
reduced by median from cloud to formula and from formula to source-law family
or frozen relation group.  The primary endpoint is family/group-level AUROC
with a 2,000-repetition grouped bootstrap interval.  No q90 or q95 alert count
is a primary endpoint.

## Execution order

1. Train the 1899-horizon model for 30 epochs, freeze its final checkpoint.
2. Evaluate and hash-freeze all Phase A internal and post-1900 outputs before
   exposing any training process to the pre-1950 increment.
3. Train a fresh cumulative pre-1900 + pre-1950 model for 30 epochs.
4. Evaluate the 1950-horizon model against its held-out known data and the
   post-1950 curated families.
5. On the identical 38 curated 1901--1950 families, report paired anomaly-score
   collapse after promotion into the admitted horizon, with a bootstrap CI and
   exact sign test.

The frozen protocol and code hashes are recorded in
`EPOCH_LEARNED_V3_PREREGISTRATION.json`.  The fail-closed remote sequence is
implemented by `run_epoch_v3_pipeline.sh`.

## Runtime profile

- GPU: NVIDIA H20 96 GB.
- Environment: PyTorch 2.14.0 + CUDA 13.0.
- Formal batch size: 1,024; point branch BF16; graph/KAN branch FP32.
- Measured Phase A steady-state throughput: approximately 5,600 clouds/s;
  peak allocated GPU memory approximately 15.4 GB.

## Paper reporting guardrails

- Say “formula-derived point clouds” rather than “measurements” for these
  temporal tests.
- Report 880,896 training clouds together with their 3,441 formula records and
  source-law/relation grouping; do not present augmentations as independent
  laws.
- “Later” is the positive evaluation stratum, not a training label learned by
  the Phase A query encoder.
- A high AUROC supports ranking transfer relative to the declared historical
  corpus.  It does not prove that every later formula is new physics, nor that
  every pre-1900 relationship belongs to the learned manifold.
- The paired score-collapse experiment is the direct epistemic-clock test:
  admitting a family should make that same family less anomalous.
