# EPOCH historical-case test objects

Status: **retrospective reconstruction — not preregistered or confirmatory**

This file makes the object of each existing case study explicit. It does not turn post-hoc choices into prospective evidence. A future sealed evaluation must instantiate the same fields before outcomes are opened and must rerun the complete adaptive procedure inside every null replicate.

## Shared decision semantics

EPOCH is one method with internally routed evidence components. Its allowed outputs are:

1. `compatible_over_measured_range`;
2. `localized_model_incompatibility` with a point estimate and calibrated interval;
3. `model_incompatibility_boundary_not_identifiable`;
4. `insufficient_evidence_abstain` with a reason code.

An alert means incompatibility with the declared incumbent model and observation process, or a measurement-process anomaly. It does not by itself identify new physics or a successor law.

## FIRAS blackbody spectrum

- Evidence type: published processed spectrum with pointwise uncertainties; 43 points.
- Route: theory-conditioned internal component.
- Incumbent model set: \(I(\nu)=A\nu^2\) (Rayleigh--Jeans shape).
- Fitted parameters: amplitude \(A\), weighted over the 43 observed points.
- Observation model: published pointwise standard errors plus an independent 0.1% response floor; frequency covariance unavailable locally.
- Statistic: sum of squared standardized residuals after refitting.
- Null: 1,000 same-grid parametric-bootstrap replicates, with amplitude refitted in every replicate.
- Observed sampled regime: \(h\nu/kT\in[1.2,11.3]\); no pure Rayleigh--Jeans interior.
- Admissible conclusion: whole-range incumbent-model incompatibility; boundary not identifiable from these data.

## Bertozzi fast electrons

- Evidence type: five values digitized from the 1964 publication.
- Route: theory-conditioned internal component.
- Incumbent model set: \(\beta^2=2KE/(m_ec^2)\), with \(m_ec^2=0.511\) MeV.
- Fitted parameters: none.
- Observation model: independent 0.02 floor in \(\beta^2\), representing digitization and measurement uncertainty.
- Statistic and null: sum of squared standardized residuals; 1,000 same-grid parametric-bootstrap replicates.
- Observed sampled regime: the lowest point is already \(\beta^2=0.752\); no non-relativistic interior.
- Admissible conclusion: whole-range Newtonian-model incompatibility; boundary not identifiable from these data.

## NIST OFHC-copper heat-capacity reference curve

- Evidence type: 120 evaluations of a NIST reference polynomial; not raw independent measurements.
- Route: data-anchored for candidate localization; theory-conditioned for the clock-advance comparison.
- Incumbent model set: constant Dulong--Petit plateau in log response.
- Fitted parameters: one log-plateau level, fitted only on the upper temperature quartile.
- Observation model: independent 5% relative/log error floor. The reference-polynomial covariance is unavailable locally.
- Statistic and null: log-space squared standardized residuals; 1,000 same-grid parametric-bootstrap replicates with the plateau refitted.
- Admissible conclusion: candidate localized incompatibility relative to the declared constant model. It is one physical mechanism and reference-curve evidence, not six independent discoveries. Boundary precision remains provisional until raw-data covariance is supplied.

## Onnes mercury resistance

- Evidence type: nine figure-digitized historical values; four sub-transition values are upper limits.
- Route: data-anchored localization candidate and theory-conditioned clock comparison.
- Incumbent model set: smooth linear normal-state resistance over the narrow observed range.
- Fitted parameters: slope and intercept, fitted above 4.21 K.
- Observation model: 0.003 ohm for uncensored points; sub-transition observations represented at a 2e-6-ohm upper limit in the current development code.
- Statistic and null: squared standardized residuals; 1,000 same-grid parametric-bootstrap replicates with the normal trend refitted.
- Admissible conclusion: candidate localized incompatibility near 4.20 K, with explicit digitization and censoring caveats. A confirmatory analysis requires a censoring-aware likelihood rather than treating limits as ordinary values.

## Millikan photoelectric relation

- Evidence type: four digitized sodium stopping-potential values.
- Route: theory-conditioned internal component; the generic shape component admits the observed line.
- Incumbent model set: frequency-independent stopping potential over the tested frequencies.
- Fitted parameters: one constant level.
- Observation model: independent 0.02 V pointwise error from the development protocol.
- Statistic and null: squared standardized residuals; 1,000 same-grid parametric-bootstrap replicates with the level refitted.
- Admissible conclusion: incompatibility with the supplied constant-energy prediction. The fitted zero crossing is extrapolated because all four retained points are above threshold; a boundary is therefore not identifiable from this reduced dataset. Instrumental alternatives and historical intensity controls remain external to the audit.

## Evidence that is not interchangeable

Raw observations, processed measurements, figure digitizations, evaluated reference curves and formula-forward synthetic point clouds are distinct evidence types. Formula-forward records are permitted in formula-level calibration and representation learning but cannot enter a real-measurement performance denominator. Repeated materials, noise realizations and algebraic variants are grouped by physical/source mechanism for inference.
