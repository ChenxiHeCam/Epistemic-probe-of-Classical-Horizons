# Auditing predictive boundaries from historically restricted physical knowledge

**Chenxi He**<sup>∗,1</sup> and **Jingqiu Chen**<sup>2</sup>

<sup>1</sup> Cavendish Laboratory, University of Cambridge, Cambridge CB3 0HE, United Kingdom
<sup>2</sup> College of Physics and Optoelectronic Engineering, Shenzhen University, Shenzhen 518060, China
<sup>∗</sup> Correspondence: ch2067@cam.ac.uk

---

## Abstract

Scientific models have finite domains. We ask whether a system restricted to knowledge available before a chosen date can recognize where that knowledge stops predicting data. EPOCH combines positive-only representation learning, data-anchored extrapolation and theory-conditioned residual tests. A 20.9-million-parameter encoder trained on 880,896 point clouds from source-audited pre-1900 relations ranks 38 formula families from 1901--1950 above held-out pre-1900 groups with AUROC 0.921 (0.833--0.994), and all 55 later families with AUROC 0.925 (0.851--0.982), compared with 0.649 before optimization. A noise-matched diagnostic gives AUROC 0.857, and an independently frozen four-form component gives 0.888 on the 55-family audit. At a threshold calibrated to 5% false positives, the latter detects 27% of synthetic departures. In historical measurements, four of five old-horizon mismatches disappear when the successful successor prediction is admitted. EPOCH tests a declared knowledge horizon: what it explains, where its predictions cease to hold, and whether a later theory removes the discrepancy.

---

## Introduction

Machine-learning systems in science are usually optimized for interpolation, forward prediction or recovery of a governing expression.<sup>1,2</sup> We ask a different diagnostic question: can a system restricted to earlier knowledge recognize when a declared model ceases to predict observations? Shape novelty alone cannot answer this question; the null prediction, observation process and decision rule must also be specified.

Several established methods address parts of this problem. Conformal prediction calibrates residuals, change-point methods locate distributional shifts, and Bayesian model discrepancy compares observations with a simulator.<sup>3–5</sup> Symbolic regression searches for a compact replacement once a discrepancy has been found.<sup>6,7</sup> None of these operations, by itself, defines which body of physical knowledge is being tested. We make the knowledge horizon an explicit input.

EPOCH uses three sources of evidence. The data-anchored component extrapolates a generic form from an incumbent regime. The theory-conditioned component tests a numerical prediction. The learned component measures distance from a positive-only representation of the historical corpus. Available scores are returned together, without averaging unlike quantities.

We set the knowledge horizon to 1899. Every admitted training relation is linked to a dated source-law family, splits are grouped by family, and learned optimization contains neither later formulas nor artificial breakdown templates. The large-scale protocol was timestamp-frozen before temporal scoring. Same-seed unoptimized and generator-matched controls isolate the ranking information contributed by pre-1900 learning.

The resulting object is predictive credibility relative to a declared model set. The test first asks whether observations remain compatible with the incumbent. If the measurements include an incumbent regime, it also estimates where compatibility is lost. This defines the starting point for successor-model search and physical interpretation.

At the 1899 horizon, the large learned component separates later formula families from pre-1900 controls with AUROC 0.925, and an independently frozen statistical component reaches 0.888. In a paired historical test, the successful successor prediction removes four of five old-horizon discrepancies. These experiments test recognition, localization and advancement of the same declared horizon.

## Results

### Declaring the object of a boundary test

Each test returns the available component scores through a common verdict interface. Scores are calibrated against their own reference distributions and are reported separately.

The data-anchored component receives a notation-free point cloud. A fixed rule searches for a contiguous interior regime, fits the best admissible generic form there and tests its extrapolation at the frontier. The theory-conditioned component instead forms measurement-model-aware residuals against an incumbent-theory prediction and tests them against a case-matched counterfactual null. EPOCH returns one of four semantic verdicts: *compatible over the measured range*; *localized model incompatibility* with an uncertainty interval; *model incompatibility over the sampled range, boundary not identifiable*; or *insufficient evidence / abstain*. Routing is fixed in advance: use data-anchored evidence when a clean interior is observed; otherwise use theory-conditioned evidence if an incumbent prediction and uncertainty model are available; otherwise abstain.

An EPOCH test is the tuple
$$
\mathcal{T}=(\mathcal{M},\Theta,P_\varepsilon,I,A,S,\alpha),
$$
where $\mathcal{M}$ is the admissible model set, $\Theta$ the fitted parameters, $P_\varepsilon$ the observation and censoring model, $I$ the fitting data, $A$ the complete adaptive procedure, $S$ the discrepancy statistic and $\alpha$ the error rate. This declaration connects every verdict to a reproducible prediction-and-measurement model. A named physical-theory conclusion follows when $\mathcal{M}$ and $P_\varepsilon$ encode that theory's relevant predictions and observation process.

The data-anchored component combines split-conformal nonconformity with an extrapolation-residual ratio from power, exponential, linear and constant fits. Formula-derived classical point clouds with measurement-level noise supply the calibration distribution. The finite-sample upper-tail probability $p=(1+\#\{S_i\geq S_*\})/(n_{cal}+1)$ supports confirmatory testing, with Benjamini--Hochberg control for dataset scans;<sup>8</sup> the 0.5 percentile remains a screening threshold. Calibration includes the complete direction, window and model-selection procedure, so the resulting guarantees apply to the declared exchangeable generator.

A source-audited reconstruction contains 268 point clouds from 51 cited pre-1900 law families, split by family into 205/17/46 training, validation and test records. A compact encoder trained on the 205 classical records, without deformation negatives, reaches family AUROC 0.903 (family-bootstrap 95% CI 0.729--1.000), compared with kNN/Mahalanobis AUROCs of 0.329/0.288 for random features. Its frozen threshold clears all 12 synthetic classical families and seven cited classical test families while detecting 2/12 breakdown families. A unit-preserving deformation audit gives source-family-balanced AUROC 0.684 with units and 0.922 without them, separating representation ranking from threshold sensitivity.

A capacity- and data-matched generic-function encoder controls for dataset size, architecture, seed, schedule and functional-signature histogram. Its family AUROC is 0.375 [0.167, 0.618], giving a paired difference of 0.528 [0.264, 0.764]. Across three paired seeds, pre-1900/generic AUROCs range over 0.889--0.910/0.292--0.403; all pre-1900 runs clear the classical controls. The gain is specific to the source-audited pre-1900 corpus rather than generic function exposure.

### Historical measurements and classical controls

The initial data-anchored screen gives percentiles of 0.95 for FIRAS, 0.95 for copper specific heat, 0.66 for Bertozzi and 0.66 for Onnes. Case-specific tests then determine whether the available measurements support whole-range rejection or localization. Supplementary Note 5 gives finite-sample tail probabilities and the multiplicity calculation.

The four examples cover two outputs: whole-range model rejection and boundary localization.

| dataset | evidence object | incumbent regime sampled? | EPOCH output |
|---|---|---:|---|
| FIRAS monopole spectrum | published processed spectrum with pointwise errors | no pure Rayleigh--Jeans regime | whole-range incumbent-model rejection |
| Bertozzi electrons | five values digitized from the published historical report | no non-relativistic regime | whole-range incumbent-model rejection |
| NIST heat capacity | evaluated reference-polynomial curves | yes for five of six materials | candidate boundary relative to the high-temperature model |
| Onnes mercury | historical figure digitization with upper-limit censoring | normal-state segment present | candidate boundary near 4.20 K |

The routing rule reports whole-range incompatibility when an incumbent prediction is available but its valid regime is not sampled, and reports *insufficient evidence* when neither route is supported.

**Quantum radiation (COBE-FIRAS).** In the 43-point cosmic-microwave-background monopole spectrum,<sup>9</sup> a Rayleigh–Jeans law proportional to ν² over-predicts measured intensity above the peak by 475-fold to 3,000-fold across fitting windows; the 20% window gives a bootstrap interval of [1,500, 2,400]. At T = 2.725 K, FIRAS samples x = hν/kT from 1.2 to 11.3 rather than the pure Rayleigh–Jeans regime x ≪ 1. The output is whole-range incompatibility with the declared classical prediction.

**Relativity (Bertozzi 1964).** Bertozzi's electron measurements<sup>10</sup> report kinetic energy against β² = v²/c². The classical law ½mv² crosses β² = 1 near 0.26 MeV and reaches β² = 58.7 at 15 MeV, equivalent to 7.7c, whereas the measurements approach β² = 1. The lowest point is already 0.87c, placing the complete series outside a non-relativistic interior. EPOCH rejects the Newtonian energy–velocity prediction over the sampled range.

**Quantum thermodynamics (specific heat of six solids).** The Dulong–Petit law predicts temperature-independent molar heat capacity, whereas the NIST cryogenic reference curves fall at low temperature. Five materials contain a high-temperature plateau and support candidate localization relative to the constant model; beryllium supports whole-range incompatibility. Interior R² ranges from 0.965 to 0.998 and departure ratios from 127 to 1,490. For OFHC copper, the classical extrapolation over-predicts the 4 K reference value by roughly 3.7×10³. The reference-polynomial curves share one low-temperature mechanism and are grouped accordingly.

**Superconductivity (Onnes 1911).** Onnes's mercury resistance values,<sup>11</sup> digitized from R(T) and anchored by the reported resistance below 10⁻⁵ Ω, show a smooth normal-state trend followed by a discontinuous drop. Relative to the declared smooth and residual-resistance models, EPOCH localizes a candidate boundary near 4.20 K. Figure-reading uncertainty and censored lower values are incorporated in the case-specific evidential status.

Three real controls span two branches of physics and three and a half centuries of measurement. Independently measured Solar-System orbits recover Kepler's exponent of 3/2 to four figures (1.500) and score 0.02. The four Galilean moons give exponent 1.500 and score 0.00. Boyle's original 1662 pressure–volume table gives exponent −0.998 and score 0.38. All three fall below the 0.5 screening threshold.

### Frozen transfer beyond the knowledge horizon

A blinded transfer evaluation contains 200 historical formula records and 30 synthetic realizations of five breakdown families, with labels hidden during scoring. The detector reaches AUROC 0.88. At the 0.5 screening threshold, recall is 0.90 and the false-positive rate is 0.24. The calibrated analysis below uses a stricter operating point.

We additionally constructed a citation-anchored temporal-transfer audit containing 55 named physical-law families first valid from 1901 to 1996 (Fig. 7b). Eight independently sampled and noisy 120-point realizations per family give 440 formula-generated point clouds. Neither component is retrained or recalibrated: the zero-unit encoder and its pre-1900 reference bank, together with the four-form statistical scorer, are frozen before these records are scored. Eleven cited pre-1900 source-law families excluded from encoder training serve as controls. All quantities are reduced to one median per source-law family.

The learned component ranks post-1900 families above held-out pre-1900 controls with AUROC 0.669 (family-bootstrap 95% CI [0.534, 0.798]); the four-form component reaches 0.888 [0.800, 0.959]. AUROC directly measures the frozen ordering of later families against earlier holdouts, with source-law families as the bootstrap unit. Supplementary Note 15 reports threshold sensitivity.

Functional structure explains part of the ranking. Learned/statistical AUROCs are 0.629/0.795 for 12 later affine, power or exponential laws, 0.600/0.982 for five threshold or piecewise families and 0.909/0.864 for six nonmonotone or oscillatory families. The two scores respond differently to smooth familiar forms, thresholds and nonmonotone curves.

We next scale the positive-only learned component to 880,896 normal-law point clouds derived from 3,441 pre-1900 formula reductions. The 20.9-million-parameter model aligns a permutation-invariant point-cloud student to a training-only formula-graph teacher. At inference, it receives only an unordered normalized point cloud—no formula, graph, units, date, source or anomaly label. Data views, checkpoint rule, score construction, aggregation and bootstrap procedure were timestamp-frozen before temporal scoring.

Against nine held-out pre-1900 historical-family groups, the 1899-horizon score reaches AUROC 0.921 [0.833, 0.994] on 38 citation-anchored 1901--1950 families, 0.935 [0.823, 1.000] on 17 post-1950 families and 0.925 [0.851, 0.982] on their 55-family union (Figs. 7a and 8). The same-seed, architecture-matched zero-optimization model reaches 0.589, 0.784 and 0.649, respectively. All four trained score components support separation on the 55-family union, with AUROCs from 0.800 to 0.925.

We then applied the frozen components to 11 additional twentieth-century cases (Fig. 7c--g). Learned/statistical coordinates include Casimir plates (1.000/0.892), Cherenkov emission (0.823/0.965), cosmic acceleration (0.908/0.429) and Pound--Rebka redshift (0.071/0.277). Representation distance and explicit extrapolation pick up different departures. Scalar or sparse-level claims, including the Lamb shift and electron anomalous magnetic moment, instead use the theory-conditioned component. The primary AUROC is unchanged (Supplementary Note 17).

A registered domain diagnostic regenerates all nine internal pre-1900 groups with the later benchmark's sampling and noise process. Against these matched controls, AUROC is 0.885 [0.778, 0.965] for the 38 near-future families and 0.857 [0.741, 0.945] for all 55 later families; zero-optimization values are 0.516 and 0.545. The retained separation shows that pre-1900 optimization contributes beyond the native generator contrast, with uncertainty governed by the nine negative families.

### Calibrated operating points define sensitivity

On the 36-dataset fair suite, the statistical detector reaches AUROC 0.80 [0.64, 0.94]. Comparison methods range from 0.58 for Gaussian-process discrepancy to 0.77 for split-conformal prediction. The anti-circularity ablation gives AUROC 1.00 when the classical law is supplied or selected automatically and 0.97 when both law and interior are selected automatically.

We calibrated the data-anchored score separately at n = 5, 9, 43 and 120 using 12 departure and 12 classical families. At n = 120, TPR is 0.271 [0.073, 0.500] at the frozen 5% quantile and 0 at the 1% quantile; realized FPR is 0/96 at both thresholds. Sensitivity is 1.0 for smooth roll-off and discontinuous-drop simulations and zero for seven families. This operating point favours specificity, and the family breakdown shows where sensitivity is gained or lost.

### Advancing the horizon collapses case-matched discrepancies

To test clock advancement, we score each dataset twice: first under its old incumbent prediction, then under the historically successful successor, using the same grid and error model. The old horizon is rejected in all five cases (bootstrap p = 0.001). The advanced horizon clears FIRAS under Planck's law (p = 1.000), Bertozzi under relativistic kinetic energy (p = 0.089), Onnes under a superconducting transition model (p = 1.000) and Millikan under the photoelectric relation (p = 0.900). Mismatch statistics fall by factors from approximately 2.0×10⁴ to 7.3×10¹⁰.

Copper heat capacity is sensitive to the error model. A Debye-plus-electronic form reduces mismatch by 3.86×10³ but remains rejected with a 5% reference-curve error floor (p = 0.001), clearing at 7.5%. Across all five cases, low-dimensional wrong-shape families remain rejected after BH correction (q ≤ 0.0015), as do dimensionally coherent but mechanism-irrelevant forms (1,000 bootstrap replicates; p = q = 1/1001). Four of the five score collapses are specific to the successful successor prediction.

### Measurement-process stress tests and boundary uncertainty

We next altered the observation process while keeping the underlying classical relation fixed. At the strongest simulated levels, a detector calibrated to 5% reports false-positive rates of 0% for calibration drift, 40% for sensor saturation, 63% for censoring, 9% for x-axis error, 11% for outliers, 2.5% for correlated noise and 0% for heteroscedasticity. The output schema therefore carries a measurement-process branch, with saturation and censoring tested explicitly for plateau and threshold cases.

Ordinary measurement bootstrap intervals cover 53.9% of true onsets at a nominal 90% level and 8.5% for steps. Size-conditional split-conformal radii calibrated on 899 independent simulations raise held-out coverage to 91.3% [89.3%, 93.0%] across 900 simulations, with family-wise coverage from 87.8% for smooth turnovers to 94.4% for kinks. The output is therefore an interval rather than a single estimated onset.

Simulations extend the evaluation into incumbent regimes absent from the FIRAS and Bertozzi measurements. Blackbody radiation spanning x = hν/kT from well below to well above unity contains a Rayleigh–Jeans interior with R² = 0.99, followed by a departure with conformal score 468 and ratio 192. Relativistic kinetic energy yields a localized boundary at 0.86c, and simulated photoelectric voltage localizes the frequency threshold. These controlled cases connect whole-range rejection to boundary recovery.

### From boundary detection to successor-law inference

Boundary detection comes before form recovery. Once EPOCH has found a calibrated mismatch, symbolic regression can search for a replacement under a parsimony prior. The Bayesian machine scientist and AI Feynman, for example, recover Planck-type expressions from blackbody-type data.<sup>6,7</sup> Physical interpretation then supplies the mechanism behind the fitted form.

Millikan's 1916 photoelectric measurement gives a concrete example.<sup>12</sup> Stopping potential is linear in frequency, V_stop = (h/e)(ν − ν₀), with R² = 0.996 and a slope that recovers h/e to 0.3% (Fig. 5). The generic dictionary accepts this linear shape. The four measurements reject the classical prediction of frequency-independent photoelectron energy. Their fitted intercept gives ν₀ = 4.39×10¹⁴ Hz; because all four frequencies lie above ν₀, the output is whole-range model rejection rather than localization of the emission threshold.

The resulting verdict starts the comparison among incumbent theory, the observation process and candidate successor models.

## Discussion

The main result is that learning from pre-1900 relations changes how later physical relations are ordered. The trained encoder reaches AUROC 0.925, compared with 0.649 for the same architecture before optimization. The effect remains at 0.857 after the pre-1900 controls are regenerated with the later benchmark's sampling and noise process. An independently frozen four-form scorer reaches 0.888. Thus, both the learned representation and explicit extrapolation contain information about departures from the 1899 knowledge horizon.

The old-to-new horizon experiment tests whether a mismatch is tied to a named incumbent model. Four of five datasets become compatible when the successful successor prediction is admitted, whereas wrong-shape and wrong-mechanism alternatives remain rejected. The copper result depends on the assumed uncertainty of the reference curve, showing that the observation model is part of the knowledge horizon rather than a nuisance added after scoring. For closed-form theories, advancing the horizon amounts to replacing the admitted prediction and recalibrating its reference distribution. Simulator-based references give the corresponding construction when closed forms are unavailable.

The three components cover different failure modes. Later affine, power and exponential laws often remain close to the generic dictionary. Threshold, piecewise, nonmonotone and oscillatory relations separate more strongly, but a physically decisive discrepancy can also retain a simple shape. The Millikan and Pound--Rebka examples illustrate the latter case: their interpretation comes from disagreement with a specified physical prediction, not from geometric novelty. Reporting the component scores separately preserves this distinction and makes the reason for each verdict inspectable.

EPOCH combines ideas from neural extrapolation,<sup>1,2</sup> split-conformal prediction, Bayesian model discrepancy and change-point detection.<sup>3–5</sup> It is also related to model-independent searches in particle physics<sup>15</sup> and to machine rediscovery of natural laws.<sup>6,7,16</sup> The difference is the target of inference. EPOCH tests the predictive domain of a declared model before asking symbolic regression or theory construction for a replacement.

This target is useful at two scales. Large libraries of measured relations can first be ranked against a frozen knowledge horizon. Selected cases can then be rerun with explicit measurement errors and incumbent predictions to obtain a model test and, where an incumbent regime is sampled, a boundary interval. The same representation and decision interface can be moved to another period by changing the dated training corpus, admitted predictions and calibration data.

### Limitations and outlook

The present validation combines a source-audited 1899 training horizon with formula-generated temporal benchmarks, simulations and retrospective historical measurements. The primary learned comparison has nine held-out pre-1900 groups. The generator-matched analysis and 11-case expansion follow the primary scoring, and the native generator-gated comparison is near chance (AUROC 0.536). The cumulative pre-1950 corpus awaits the same source-level audit. The data-anchored vocabulary contains four generic forms, and boundary coverage is calibrated under the stated simulators. Saturation and censoring are the main measurement-process confounders found here. A prospective benchmark should freeze the full decision path and use mechanism-independent measurements with audited instrument models. These additions test generality; they do not alter the reported 1899-horizon experiment.

## Methods

**Dictionary.** The data-anchored component uses a fixed parsimonious vocabulary of power, exponential, linear and constant forms. An eight-form ablation adds quadratic, cubic, square-root and logarithmic terms; the four-form version gives the stronger validation extrapolation performance and is used throughout.

**Interior and frontier split.** A fixed rule defines the candidate interior as the contiguous regime in which one admitted generic form fits with high internal R²; the frontier is the complementary region. Fit quality selects the low- or high-value direction. Every calibration replicate reruns the complete direction search, candidate windows, model comparison, parameter fitting, stopping rule and final score.

**Statistical detector and decision rules.** Split-conformal nonconformity is computed from interior residual quantiles. The extrapolation-residual ratio compares median log-residual at the frontier with its interior value. Each form is fitted from multiple initializations, with a one-percent relative floor in both denominators. The reproduced legacy score averages rescaled components clipped at 3. Confirmatory tests use the finite-sample upper-tail probability $p=(1+\#\{S_i\geq S_*\})/(n_{cal}+1)$, including ties. The primary endpoint is TPR at a threshold targeted to 5% FPR on independent calibration; TPR at 1% FPR is secondary. Dataset scans report BH-adjusted q-values, and relations below 20 points use size-matched calibration. Exchangeability is defined at the source-law family or mechanism-family unit, with the complete selection pipeline included in calibration.

**Learned component.** The compact set transformer (d = 128, three induced-set-attention blocks) is trained for 1,000 steps on two independently subsampled and rescaled views of 205 source-audited classical point clouds from 40 families. Sampling families without replacement prevents repeated laws from becoming false contrastive negatives, and an auxiliary head predicts functional signatures. Matched models use SI dimensions or zeroed dimension metadata; the zero-unit model is primary for the synthetic benchmark. Encoder optimization uses no breakdown, saturation, step, kink, splice or crossover. Frozen one-class kNN and shrinkage-Mahalanobis scores are compared with a matched random encoder. Cross-conformal calibration removes each training family in full, reduces eight measurement views within records and then to one median per family. The higher-order 95th-percentile statistic over 40 leave-family-out family scores is frozen before evaluation. Hash-linked training and evaluation files enforce checkpoint/data identity.

**Large-scale graph-aligned pre-1900 representation audit.** A 20,908,547-parameter dual encoder tests scaling without anomaly supervision. Its query branch is a permutation-invariant six-block Set Transformer<sup>13</sup> (width 384, eight heads, four pooling tokens and a 256-dimensional normalized output). During positive-only pretraining, it is aligned to a formula-structure teacher with four edge-aware graph updates and two graph-local attention layers. One graph-update MLP uses a 96-channel, eight-basis RBF-KAN module.<sup>14</sup> The frozen query receives a median/IQR-normalized unordered ((x,y)) cloud, with no equation, graph, units, date, law name, post-cutoff label or deformation example.

Phase A uses 880,896 normal-law clouds derived from 3,441 audited pre-1900 formula reductions, grouped into 91 contrastive identities spanning 64 historical-family tags. Held-out calibration and internal sets contain 11,648/62,336 clouds from 91/487 formulas. After scientific-group reduction, nine historical-family groups form the internal AUROC control class. Citation-anchored temporal tests contain eight measurement views for each of 38 families dated 1901--1950 and 17 later families. A separate 96,064-cloud generator-gated set and the cumulative pre-1950 pilot are reported in Supplementary Note 16.

The pre-1900 model trains for 30 epochs at batch size 1,024 under a checkpoint rule fixed before temporal evaluation. The one-class score averages four calibrated ranks: top-eight cosine distance to graph-formula embeddings, top-eight distance to identity centroids, top-eight distance to point-formula prototypes and shrinkage-Mahalanobis distance to those prototypes. Each component is transformed by its pre-1900 calibration empirical CDF; clouds are reduced by median to formulas and then to source-law families or frozen relation groups. Family/group-level AUROC uses 2,000 grouped-bootstrap replicates. An architecture-matched, same-seed, zero-optimization checkpoint runs through the same pipeline. Code, configuration, score construction, seed and checkpoint rule were timestamp-frozen before temporal scoring.

A timestamped compatibility addendum replaced the unavailable call `torch.flatnonzero` with the equivalent public operation `torch.nonzero(..., as_tuple=False).flatten()` before any score was written. For the generator-matched analysis, all nine internal control families were regenerated with the temporal benchmark's point count, x-jitter, repeat count, noise range and heteroscedastic profile. The checkpoint, reference memories, calibration CDF, score and aggregation were unchanged; the same analysis was run with the zero-optimization checkpoint.

**Matched generic-function control.** A second zero-unit encoder uses the same architecture, seed, batch size, augmentation, auxiliary objective and 1,000-step schedule. Its 205 point clouds come from globally valid smooth expressions in 40 pseudo-families. Record count, family-size profile and the histogram of eight-bit functional signatures match the pre-1900 training set exactly. Calibration uses the same leave-family-out reduction.

**Training-seed sensitivity.** Both encoders are retrained from scratch with three paired initialization/augmentation seeds while the evaluation-view seed and 24 mechanism families remain fixed. We report the observed range and sample standard deviation across these runs; canonical checkpoints are unchanged.

**Unit-metadata-preserving deformation audit.** Each of 46 held-out point clouds is paired with saturation, step, kink, splice, roll-off and crossover deformations that retain its archived input/output SI vector. Scores are reduced to one clean and six deformation values within each of seven source-law families; the AUROC difference is bootstrapped by family. Each checkpoint uses its own higher-order 95th-percentile threshold from the same 40 leave-family-out training units. Thirty-four records have at least one nonzero dimensional exponent.

**Learned-component routing.** Point clouds with at least 20 observations use the learned score. Near-constant responses route to a theory-conditioned constant/noise model. Formula and deformation families are the resampling units. The deformation-supervised comparison uses anomaly-shape priors, whereas the primary positive-only encoders use normal pre-1900 relations alone.

**Frozen post-1900 formula transfer.** The registry contains 55 source-law families first valid from 1901 through 1996, each with a primary historical citation; 45 include a DOI. Each law is represented by a disclosed one-dimensional nondimensional reduction over a fixed support, with eight 120-point realizations varying nuisance parameters, sampling and 0.5--2.5% noise. The corpus contains 440 records and 55 inferential units. Special reductions, including the BCS interpolation, two-flavour neutrino survival curve and ideal quantum-Hall staircase, are marked in the registry. Offline checks cover family and seed uniqueness, citation-year consistency, finite nonconstant curves, point ordering, file hashes and family separation from the pre-1900 archive.

No training or calibration is performed during transfer. The checkpoint and pre-1900 data hash must match the cross-family audit. The learned score is reduced by median over measurement views, records and families; its threshold is the pre-existing higher-order q95 statistic over 40 leave-family-out calibration families. The statistical score reruns the full four-form selection and extrapolation procedure with the existing n = 120 thresholds. Eleven cited pre-1900 families serve as controls, and AUROC intervals resample families. Twelve post-1900 laws overlap the declared affine, power or exponential forms. Per-family conformal p-values and component union/intersection rates are reported as secondary diagnostics.

**Synthetic suites.** The legacy held-out evaluation comprises 180 labelled datasets and the fair baseline suite comprises 36. The strict development benchmark contains 12 classical and 12 breakdown families with eight realizations per family. Its 5% and 1% thresholds are estimated from 785 valid, size-matched calibration simulations; uncertainty is resampled by mechanism family rather than treating noise realizations as independent phenomena. The learned checkpoint audit uses 192 datasets spanning 12 classical and 12 breakdown families.

**Real data and preprocessing.** We analyse the published COBE-FIRAS monopole spectrum and pointwise errors; five Bertozzi values digitized from the 1964 report; NIST specific-heat reference polynomials for six materials; Onnes resistance values, including upper limits; Millikan sodium stopping potentials; and Solar-System orbital constants. Each serialized test object records input status, model set, fitted parameters, observation model, fit subset, adaptive procedure, decision rule, output and provenance. Old- and advanced-horizon counterfactual nulls use the same x-grid. Explicit error floors represent unavailable covariance and digitization uncertainty and are varied in sensitivity analysis.

**Horizon-specificity controls.** Wrong-shape and dimensionally coherent wrong-mechanism forms are refitted on the same case grids and observation models with 1,000 parametric-bootstrap replicates. The physical controls substitute fermionic occupation for photon statistics, a proton for the electron rest-energy scale, an Einstein oscillator for Debye acoustic modes, Arrhenius transport for the superconducting transition and Compton recoil for the photoelectric slope. Dimensional ratios and parameter units are retained in the machine-readable output.

**Baselines, robustness and localization.** Split-conformal prediction, the extrapolation-residual ratio, Gaussian-process discrepancy, CUSUM and RESET are evaluated on the fair suite. Instrument tests apply calibration drift, sensor saturation, censoring, x-axis error, outliers, correlated noise and heteroscedasticity to classical synthetic relations. Boundary intervals use split-conformal calibration of normalized localization error separately at n = 40, 80 and 160.

**Real-measurement inventory.** We downloaded 44 files from the NIST Dataplot and Statistical Reference Dataset collections and froze their SHA-256 hashes. The inventory contains 20 instrument controls, 11 classical controls, six modern-known controls and seven horizon cases. Promotion to the evaluation registry requires audited column semantics, incumbent prediction, measurement model, provenance and mechanism-level grouping.

## Figure legends

![](figures/Figure_1_concept_revised.png)

**Figure 1 | Testing a declared knowledge horizon. a**, Measurements follow an incumbent description and then depart from its extrapolation. The test measures the discrepancy and localizes its onset. **b**, EPOCH routes each case according to the available evidence: data-anchored extrapolation from a sampled incumbent regime, or a theory-conditioned test of a numerical prediction. **c**, In the clock-advance experiment, the same measurements are scored under old and advanced horizons. Admission of the successful successor law should collapse the old-horizon discrepancy.

![](figures/figure_breakdowns.png)

**Figure 2 | Historical measurements under the 1899 horizon.** Black points show published processed measurements, digitized historical values or evaluated reference curves as labelled; blue curves show incumbent predictions. Orange badges give screening percentiles. FIRAS and Bertozzi yield whole-range model rejection. Specific heat and Onnes contain an incumbent segment, allowing a candidate boundary to be marked by the shaded frontier. Tail probabilities and multiplicity calculations are reported in Supplementary Note 5.

![](figures/figure_controls.png)

**Figure 3 | Classical controls, synthetic benchmark and learned ablations. a–c**, Solar-System orbits, Galilean moons and Boyle's pressure–volume table recover their expected exponents and remain below the screening threshold. **d**, AUROC on the fair synthetic suite (n = 36). **e**, On the 180-dataset suite, the deformation-supervised probe reaches AUROC 0.83 versus 0.62 for its statistical comparator; the positive-only probe reaches 0.453 versus 0.695. The difference measures the contribution of morphology supervision.

![](figures/figure_rigor.png)

**Figure 4 | Legacy synthetic calibration, anti-circularity and point localization. a**, ROC on the 36-dataset fair suite, with a bootstrap 95% confidence band. Its reported 62% TPR at 5% FPR is a legacy result; the broader mechanism-level audit in Fig. 6 gives 27%. **b**, Anti-circularity ablation. AUROC is 1.00 when the classical law is given, 1.00 when selected automatically and 0.97 when both law and interior are selected automatically. **c**, Point localization error on synthetic onsets. Calibrated interval coverage is evaluated separately in Fig. 6c.

![](figures/figure_limit.png)

**Figure 5 | Simple shape, decisive theory mismatch.** In Millikan's 1916 digitized values (black points), stopping potential rises linearly with frequency as V_stop = (h/e)(ν − ν₀). The slope recovers h/e to 0.3%, and the marker denotes the extrapolated zero crossing ν₀. The generic shape score clears the line, whereas the theory-conditioned component rejects the classical prediction of frequency-independent energy (dashed line). This case motivates the explicit separation between shape and incumbent-theory evidence.

![](figures/figure_revision_evidence.png)

**Figure 6 | Calibration and controls for the EPOCH components. a**, Same-grid goodness-of-fit p-values under old and advanced horizons. Four of five advanced predictions clear at 5%; five wrong-mechanism forms remain rejected. **b**, False-positive rates under simulated instrument effects, with the largest changes caused by saturation and censoring. **c**, Held-out coverage of size-conditional 90% split-conformal boundary intervals. **d**, Positive-only learned audit after source reconstruction. The zero-unit encoder reaches synthetic AUROC 0.854 by kNN and 0.895 by shrinkage Mahalanobis; family-grouped AUROC is 0.903 [0.729, 1.000], compared with 0.329 and 0.288 for matched random features. Training uses 205 records from 40 pre-1900 families and no breakdown examples. At the frozen family threshold, it detects 2/12 breakdown families while clearing all classical controls.

The matched generic-function model in panel d uses the same synthetic evaluation families. Its family AUROC is 0.375 [0.167, 0.618], and the paired pre-1900-minus-generic difference is 0.528 [0.264, 0.764]. Across three paired seeds, pre-1900/generic AUROCs range over 0.889–0.910/0.292–0.403; pre-1900 breakdown-family detection ranges from 1/12 to 8/12 with zero classical-family alerts.

![](figures/figure_case_study_composite.png)

**Figure 7 | Temporal transfer and component scores across later physics. a**, ROC curves for the 20.9-million-parameter encoder trained on pre-1900 point clouds (blue) and the same architecture before optimization (grey), comparing 55 later formula families with nine pre-1900 groups. **b**, Independent audit of the data-anchored component (orange) and compact learned encoder (blue) against 11 cited pre-1900 families. **c**, Learned scores (L, circles) and size-matched statistical percentiles (S, squares) for 11 development cases. Triangles mark cases routed to an incumbent-conditioned test (T). **d--g**, Point clouds, later relations and incumbent comparators for Cherenkov emission, Fano resonance, cosmic acceleration and Pound--Rebka redshift. Checkpoints and calibration objects remain fixed throughout.

![](figures/figure_learned_pre1900_v3.png)

**Figure 8 | Large-scale representation audit at the 1899 horizon. a**, The frozen model separates 38 families dated 1901--1950 and all 55 later families from nine pre-1900 groups. All-later AUROC is 0.857 [0.741, 0.945] against generator-matched controls. **b**, One-class score distributions for native and matched pre-1900 controls and the two later-era strata. **c**, AUROCs for trained and same-seed zero-optimization models. The native generator-gated comparison is shown separately. **d**, Each of the four distance components and their fixed equal-weight mean improves over zero optimization on the 55-family comparison. Inferential units are source-law families or frozen relation groups.

*Figures were produced with matplotlib using a colour-blind-safe (Okabe–Ito) palette and sans-serif type, and are available as editable vector files.*

## Table 1 | Principal quantitative results

| Analysis | Result |
|---|---|
| Large learned model, 38 families from 1901--1950 vs nine pre-1900 groups | AUROC 0.921 [0.833, 0.994]; unoptimized 0.589 |
| Large learned model, 55 families from 1901--1996 vs nine pre-1900 groups | AUROC 0.925 [0.851, 0.982]; unoptimized 0.649 |
| Generator-matched learned analysis, 55 later vs nine pre-1900 groups | AUROC 0.857 [0.741, 0.945]; unoptimized 0.545 |
| Independently frozen four-form analysis, 55 later vs 11 pre-1900 families | AUROC 0.888 [0.800, 0.959] |
| Source-audited compact encoder vs matched generic-function encoder | family AUROC 0.903 [0.729, 1.000] vs 0.375 [0.167, 0.618] |
| Blinded transfer, 200 classical records and 30 synthetic departures | AUROC 0.88 |
| Calibrated data-anchored decision, 12 departure and 12 classical families | TPR 0.271 [0.073, 0.500]; 0/96 false positives at the 5%-targeted threshold |
| Clock advancement on identical data grids | 4/5 old-horizon mismatches clear under the successful successor prediction |
| Horizon-specificity controls | 5/5 wrong-shape and 5/5 wrong-mechanism predictions remain rejected |
| Boundary localization on 900 held-out simulations | 90% interval coverage 91.3% [89.3%, 93.0%] |
| Observation-process stress test at maximum severity | FPR 0.40 for saturation; 0.63 for censoring; ≤0.11 for all other tested effects |

## References

1. Wang, H. et al. Scientific discovery in the age of artificial intelligence. *Nature* **620**, 47–60 (2023).
2. Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K. & Jegelka, S. How neural networks extrapolate: from feedforward to graph neural networks. *International Conference on Learning Representations* (2021).
3. Angelopoulos, A. N. & Bates, S. Conformal prediction: a gentle introduction. *Found. Trends Mach. Learn.* **16**, 494–591 (2023).
4. Page, E. S. Continuous inspection schemes. *Biometrika* **41**, 100–115 (1954).
5. Kennedy, M. C. & O'Hagan, A. Bayesian calibration of computer models. *J. R. Stat. Soc. B* **63**, 425–464 (2001).
6. Guimerà, R. et al. A Bayesian machine scientist to aid in the solution of challenging scientific problems. *Sci. Adv.* **6**, eaav6971 (2020).
7. Udrescu, S.-M. & Tegmark, M. AI Feynman: a physics-inspired method for symbolic regression. *Sci. Adv.* **6**, eaay2631 (2020).
8. Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J. R. Stat. Soc. B* **57**, 289–300 (1995).
9. Fixsen, D. J. et al. The cosmic microwave background spectrum from the full COBE FIRAS data set. *Astrophys. J.* **473**, 576–587 (1996).
10. Bertozzi, W. Speed and kinetic energy of relativistic electrons. *Am. J. Phys.* **32**, 551–555 (1964).
11. Kamerlingh Onnes, H. Further experiments with liquid helium. G. On the electrical resistance of pure metals, etc. VI. On the sudden change in the rate at which the resistance of mercury disappears. *Commun. Phys. Lab. Univ. Leiden* **124c** (1911).
12. Millikan, R. A. A direct photoelectric determination of Planck's “h”. *Phys. Rev.* **7**, 355–388 (1916).
13. Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S. & Teh, Y. W. Set Transformer: a framework for attention-based permutation-invariant neural networks. *Proc. Mach. Learn. Res.* **97**, 3744–3753 (2019).
14. Liu, Z. et al. KAN: Kolmogorov–Arnold networks. *International Conference on Learning Representations* (2025).
15. D'Agnolo, R. T. & Wulzer, A. Learning new physics from a machine. *Phys. Rev. D* **99**, 015014 (2019).
16. Schmidt, M. & Lipson, H. Distilling free-form natural laws from experimental data. *Science* **324**, 81–85 (2009).

## Data availability

The public COBE-FIRAS, Solar-System, Bertozzi, NIST, Millikan and Kamerlingh Onnes inputs are identified in the Supplementary Information. The data package contains source URLs and hashes; the 55-family formula registry and 440 generated point clouds; the 11-family development registry and 88 generated records; the pre-1900 training, calibration and evaluation manifests; and every numerical result used in the figures. The project repository is available at https://github.com/ChenxiHeCam/Epistemic-probe-of-Classical-Horizons, and the public archive is available at https://doi.org/10.5281/zenodo.22138621. The exact submission release will be deposited as a new version of the archive.

## Code availability

Code is available at https://github.com/ChenxiHeCam/Epistemic-probe-of-Classical-Horizons. The repository contains corpus audits, point-cloud generators, frozen evaluators, statistical calibration and robustness experiments, model definitions, training configurations, figure scripts and the evidence-manifest validator. Checkpoint and data hashes are verified before evaluation; the publication release will be identified by a permanent archive version and Git commit.

## Acknowledgements

We thank NASA LAMBDA, the NIST Cryogenic Material Properties database and the KNAW digital archive for public data access. This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Author contributions

C.H. conceived the framework, designed and built EPOCH, ran the experiments, and wrote the manuscript. J.C. contributed to the benchmark construction, case-study analysis, and manuscript revision. Both authors reviewed and approved the final manuscript.

## Competing interests

The authors declare no competing interests.

**Correspondence and requests for materials** should be addressed to C.H. (ch2067@cam.ac.uk).
