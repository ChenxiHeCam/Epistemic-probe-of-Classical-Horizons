# Frozen learned model

This directory holds the query-time weights of the learned component of
EPOCH, the neural encoder trained only on point clouds from relations recorded
before the 1899 knowledge horizon. Frozen means fixed, and recorded with a time
stamp and file hash, before any test data were scored.

`epoch_1899_point_student.pt` is the query-time checkpoint of the large
encoder: the 9,868,672-parameter point student taken from the frozen
20,908,547-parameter dual encoder. `epoch_1899_inference_memory.pt` holds the
pre-1900 reference embeddings, the calibration empirical distributions and the
mean and precision matrix of the shrinkage-Mahalanobis distance. The learned
score is computed from these two files: it is the distance of a new point
cloud's embedding from the embeddings of the pre-1900 training relations,
expressed as a rank on the pre-1900 calibration distributions.

At query time the model receives one normalized, unordered `(x, y)` point
cloud, the measured pairs with formula, units, labels and ordering removed.
Formula graphs, dates, source names, units and anomaly labels are training-only
inputs; the formula-graph teacher is used during pre-1900 pretraining and is
absent from this checkpoint.

| Asset | SHA-256 |
|---|---|
| `epoch_1899_point_student.pt` | `7c93e459fff4c1e4a64de9e64e4bf1299155bbd2bf3677e52d07ffef0bf03436` |
| `epoch_1899_inference_memory.pt` | `f4408c55c36329facc0b438446947aa40547c1ff66f9c43819483a8c6df60db5` |

The full 268 MB training checkpoint also contains the formula teacher,
optimizer, scheduler and contrastive queue. It is preserved in the versioned
research archive listed in the main README.
