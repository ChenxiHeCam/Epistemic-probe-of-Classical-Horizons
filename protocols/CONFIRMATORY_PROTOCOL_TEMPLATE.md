# EPOCH confirmatory evaluation protocol — template

Status: **UNFROZEN TEMPLATE — not a preregistration**

This document must be completed, version-controlled, timestamped and archived before labels or outcomes of the sealed test set are opened. EPOCH is one framework; the router selects an internal evidence component and does not define separate methods.

## 1. Primary claim and unit of analysis

- Primary claim (must be phrased as compatibility with the declared test object, not “new physics detected”):
- Knowledge-horizon date/domain:
- Admissible model set \(\mathcal{M}\), including an argument that it encodes the relevant incumbent prediction:
- Parameters \(\Theta\) permitted to be fitted and parameters held fixed:
- Observation, covariance and censoring model \(P_\varepsilon\):
- Data subset \(I\) eligible for fitting:
- Complete adaptive algorithm \(A\):
- Final statistic \(S\) and declared error rate \(\alpha\):
- Unit counted in the denominator: independent physical mechanism/event, not material, algebraic variant or noise realization.
- Primary endpoint: TPR at a calibration threshold with FPR ≤ 5%.
- Secondary endpoint: TPR at FPR ≤ 1%.
- Multiplicity: BH-FDR at 5% over all opened test mechanisms.
- Confidence interval: cluster bootstrap over mechanism groups.

## 2. Frozen router

For each dataset, routing metadata must be supplied without viewing its outcome label.

1. Use the data-anchored component only if a contiguous measured incumbent regime and admissible dictionary are declared.
2. Otherwise use the theory-conditioned component only if an incumbent numerical prediction and measurement model are declared.
3. Otherwise return `abstain`; abstentions are reported and cannot be silently excluded.

The route, direction of extrapolation, interior-selection rule and any censoring rule are frozen per dataset before scoring.

The shared output vocabulary is exactly:

1. `compatible_over_measured_range`;
2. `localized_model_incompatibility` plus point estimate and interval;
3. `model_incompatibility_boundary_not_identifiable`;
4. `insufficient_evidence_abstain` with reason code.

The third output is mandatory when the old model is rejected but no valid incumbent regime is sampled. It cannot be reported as a localized boundary.

## 3. Calibration and scores

- Score implementation and repository commit:
- Full file hash:
- Calibration generator/corpus version and hash:
- Sample-size strata:
- q95 thresholds by stratum:
- q99 thresholds by stratum:
- Tail p-value: `(1 + count(S_cal >= S_test)) / (n_cal + 1)`, ties included.
- Missing/failed fits: specify handling before evaluation.
- No threshold or score transformation may be changed after opening the sealed set.
- Calibration replicate implementation: every replicate reruns direction selection, candidate-window search, model selection, nuisance-parameter fitting, stopping/abstention and final scoring. Conditional residual calibration after selecting a winner is not accepted as calibration of the EPOCH decision.
- Exchangeability unit and justification:
- Anticipated covariate shift between fit region and frontier:
- Scope of conformal statement: marginal over the declared generator/corpus only; no automatic arbitrary-extrapolation guarantee.

## 4. Theory-conditioned nulls

For every routed dataset freeze:

- incumbent prediction and source/date;
- allowed nuisance parameters and fitting region;
- x-grid, units and transformations;
- pointwise error, covariance, calibration drift, censoring/detection limits and digitization error;
- simulator and random seed;
- number of bootstrap replicates;
- admissible irrelevant-theory negative controls.

Successor theories are inaccessible until old-horizon verdicts are deposited. Clock advancement is a second, explicitly separated paired analysis.

## 5. Boundary output

- Point-localization algorithm:
- Coverage target: 90% (95% secondary).
- Calibration distribution and sample-size radii:
- Maximum informative interval width:
- Abstain if the estimated interval exceeds that width or no stable change point is found.
- Coverage is reported overall and by mechanism, sampling density, noise and transition smoothness.

## 6. Dataset manifest

For each file record: stable ID, source URL/DOI, byte hash, acquisition date, evidence type (`raw_measurement`, `processed_measurement`, `digitized_historical_figure`, `reference_fit_curve` or `formula_forward_evaluation`), column semantics, units, mechanism group, instrument, incumbent-horizon source, error model, route and exclusion status. Outcome labels are stored in a separately sealed manifest. Formula-forward evaluations may enter a formula-level benchmark but never a real-measurement denominator.

Exclusions allowed after unsealing: only file corruption or a prespecified input-validity failure. Every exclusion and abstention remains in the flow diagram.

## 7. Learned component

- Training records require source citation and earliest-valid date.
- Splits are grouped by source law and structural family.
- Exact/algebraic duplicates cannot cross splits.
- Primary classical-only score receives no breakdown examples.
- Deformation-supervised results are labelled separately as morphology supervision.
- Units/no-units comparisons use variables with audited physical dimensions.
- The four historical development cases cannot be used for tuning.

## 8. Required reporting

- Full confusion table at 5% and 1% FPR.
- Frozen threshold, calibration target FPR and independently realized test FPR reported as different quantities.
- Finite-sample p and BH q for every dataset, including failures.
- Mechanism-clustered confidence intervals.
- Abstention and fit-failure rates.
- Instrument hard-negative results.
- Old, irrelevant-update and correct-successor paired results.
- Boundary coverage and interval width.
- Deviations from this protocol, with timestamps, reported before interpretation.

## 9. Freeze record

- UTC timestamp:
- Repository commit:
- Archive DOI/hash:
- Protocol SHA-256:
- Calibration manifest SHA-256:
- Test-file manifest SHA-256 (without outcomes):
- Outcome custodian and release procedure:
- Authors' signatures:

Until every field above is completed and archived before outcome release, the evaluation is retrospective/developmental and must not be called preregistered or sealed.
