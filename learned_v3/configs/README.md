# EPOCH two-horizon run configurations

These files fix the data protocol before model training; they are frozen,
that is recorded with a time stamp and hash before any score was computed. A
knowledge horizon is the calendar date before which a relation must have been
recorded to enter training; the two configurations here use 1899 and 1950.

- `train_pre1900.json` trains on relations recorded through 1899 and keeps
  every pre-1950 cloud out of optimization.
- `train_pre1950_joint.json` adds the pre-1950 training groups that exist
  only in that period, and is run only after the first model's post-1900
  results have been frozen.
- `test_both.json` defines the paired evaluation of the two checkpoints and
  the comparison in which the horizon is advanced from 1899 to 1950: the same
  families are scored by both checkpoints, and the scores of families from
  1901 to 1950 are expected to fall once those families lie inside the
  horizon.

The `strict` point-cloud profile uses pre-1900 expressions audited against the
source registry. The pre-1950 increment is generator-gated: its formulas are
drawn from the internal archive by the same generator as the pre-1900 corpus,
and it stays a candidate set until its own dated source registry is completed.
`expanded` counts are supplied only for a separately named sensitivity and
scale ablation.

Data-loader smoke examples, recorded for the remote host and its `scripts/`
layout; in this checkout the corresponding loader files are under
`learned_v3/`:

```bash
python scripts/epoch_pointcloud_dataset.py \
  --bundle pointcloud_full_final --view-index pointcloud_views \
  --view pre1900_temporal_eval --role train --provenance-mode strict

python scripts/epoch_pointcloud_dataset.py \
  --bundle pointcloud_full_final --view-index pointcloud_views \
  --view pre1950_joint --role train --provenance-mode strict

python scripts/epoch_formula_graph_dataset.py \
  --bundle formula_graphs --view pre1900_temporal_eval \
  --role train --provenance-mode strict --horizon pre1900

python scripts/epoch_symbolic_dataset.py \
  --bundle symbolic_pretrain --direct-bundle symbolic_direct \
  --chain-bundle symbolic_chain_core --view pre1950_joint \
  --role train --profile strict
```

The paired point-cloud/graph loader joins on the immutable `formula_index`.
Repeated formulas within a cloud minibatch are graph-encoded once and mapped
back through `sample_to_graph`.

Neither headline configuration enables deformation, kink, step, saturation or
crossover labels; both train on pre-horizon relations alone.
