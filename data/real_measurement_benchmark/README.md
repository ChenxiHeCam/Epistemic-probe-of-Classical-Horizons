# NIST real-measurement development inventory

This directory contains 44 files acquired from the official NIST Dataplot / Statistical Reference Dataset collection. It is an **unlabelled development inventory**, not a prospective or sealed benchmark.

- `manifest.json`: source descriptions, mechanism groups and acquisition roles. Roles are candidate uses, not anomaly labels.
- `download_nist.py`: reproducible downloader.
- `download_report.json`: source URL, byte count, SHA-256 and transfer status.
- `raw/`: unmodified downloaded `.DAT` files.
- `audit_downloads.py`: structural inventory only; it does not score EPOCH or assign physics labels.
- `inventory_report.json`: numeric-block structure, hashes and role/group counts.

Before a dataset can enter a performance denominator, an independent audit must freeze its variable semantics, units, incumbent knowledge horizon, prediction, uncertainty/covariance and censoring model, route through the unified EPOCH framework and mechanism-level dependence group. Single-column series use sample index only as a provisional abscissa; this is not a scientific variable assignment.

The collection currently contains 20 candidate instrument hard controls, 11 candidate classical controls, six modern-known controls and seven candidate horizon/post-classical cases. These counts must not be reported as tested negatives or positives.
