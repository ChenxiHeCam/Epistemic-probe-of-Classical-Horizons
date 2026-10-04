# NIST real-measurement development inventory

This directory contains 44 files acquired from the official NIST Dataplot / Statistical Reference Dataset collection. It is a development inventory: each file carries an acquisition role and a mechanism group and no anomaly label, and the sixteen series scored in the paper were promoted from it after the audit described below.

- `manifest.json`: source descriptions, mechanism groups and acquisition roles. A mechanism group is the set of series produced by one physical mechanism; a role is the candidate use assigned at acquisition.
- `download_nist.py`: reproducible downloader.
- `download_report.json`: source URL, byte count, SHA-256 and transfer status.
- `raw/`: unmodified downloaded `.DAT` files.
- `audit_downloads.py`: structural inventory of the numeric blocks in each file.
- `inventory_report.json`: numeric-block structure, hashes and role and group counts.

A dataset is promoted into the evaluation registry once an independent audit has frozen its column semantics and units, its incumbent (the pre-1899 law tested against it) and that law's prediction, its measurement model (uncertainty or covariance, and censoring), its route through the EPOCH components and its mechanism group. Frozen means recorded with a time stamp and hash before any score is computed. For a single-column series the sample index is the provisional abscissa until the audit assigns a variable.

The collection holds 20 instrument controls, 11 classical controls, six controls whose post-1900 mechanism is known and seven horizon cases. These are acquisition roles. The series tested in the paper are the sixteen promoted by `engine/build_real_measurement_cases.py` into `data/real_measurement_case_registry.json`.
