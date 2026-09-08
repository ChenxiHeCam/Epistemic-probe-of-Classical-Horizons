# EPOCH two-horizon run configurations

These files freeze the data protocol before model training.

- `train_pre1900.json` trains through 1899 and keeps every pre-1950 cloud out
  of optimization.
- `train_pre1950_joint.json` promotes the time-unique pre-1950 training groups
  only after the first model's temporal results have been frozen.
- `test_both.json` defines the paired two-checkpoint evaluation and the
  epistemic-clock score-collapse comparison.

The `strict` point-cloud profile uses registry-audited pre-1900 expressions.
The pre-1950 increment remains a generator-gated temporal candidate until its
own dated source registry is completed.  `expanded` counts are supplied only
for a separately named sensitivity/scale ablation.

Data-loader smoke examples on the remote host:

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

No deformation, kink, step, saturation, or crossover labels are enabled by
either headline configuration.
