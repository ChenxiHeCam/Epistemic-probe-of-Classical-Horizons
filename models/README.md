# Frozen learned model

`epoch_1899_point_student.pt` is the compact query-time checkpoint used by the
current EPOCH learned component. It contains the 9,868,672-parameter point
student from the frozen 20,908,547-parameter dual encoder. The accompanying
`epoch_1899_inference_memory.pt` contains the pre-1900 reference embeddings,
calibration empirical distributions and shrinkage-Gaussian state required by
the one-class score.

At query time the model receives only a normalized unordered `(x, y)` point
cloud. Formula graphs, dates, source names, units and anomaly labels are not
query inputs. The formula-graph teacher is used only during pre-1900
pretraining.

| Asset | SHA-256 |
|---|---|
| `epoch_1899_point_student.pt` | `7c93e459fff4c1e4a64de9e64e4bf1299155bbd2bf3677e52d07ffef0bf03436` |
| `epoch_1899_inference_memory.pt` | `f4408c55c36329facc0b438446947aa40547c1ff66f9c43819483a8c6df60db5` |

The full 268 MB training checkpoint additionally contains the formula teacher,
optimizer, scheduler and contrastive queue. It is preserved in the versioned
research archive rather than duplicated as a Git blob.
