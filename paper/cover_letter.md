# Cover letter

Dear Editors,

Please consider our manuscript, “Auditing predictive boundaries from historically restricted physical knowledge,” for publication in *Nature Communications*.

Scientific machine learning is usually evaluated by interpolation, prediction or recovery of a governing equation. We study a prior question: can a system restricted to knowledge available before a specified date identify where that knowledge stops predicting observations? EPOCH treats this as a test of a declared model set, observation process and error rate. It combines positive-only representation learning, data-anchored extrapolation and incumbent-conditioned residual tests, and reports whether the model remains compatible, fails across the sampled range or has a localized predictive boundary.

The central result is a source-audited temporal-transfer experiment. A 20.9-million-parameter graph-aligned point-cloud encoder is trained without anomaly templates on 880,896 point clouds derived from pre-1900 relations. It ranks 38 physical-law families from 1901–1950 above held-out pre-1900 groups with AUROC 0.921 [0.833, 0.994], and all 55 later families with AUROC 0.925 [0.851, 0.982]. The same architecture before optimization reaches 0.649. After matching the sampling and noise process across periods, AUROC remains 0.857. An independently frozen four-form extrapolation component reaches 0.888 [0.800, 0.959].

The historical experiments test whether this signal is tied to the knowledge horizon. Each dataset is scored under its old incumbent prediction and then under the successful successor, on the same grid and with the same error model. Four of five mismatches disappear after the horizon is advanced, while wrong-shape and wrong-mechanism alternatives remain rejected. Size-conditional boundary intervals achieve 91.3% coverage at 90% nominal coverage on held-out simulations. Additional cases, including Cherenkov emission, Fano resonance, cosmic acceleration and Pound–Rebka redshift, show how the learned, extrapolation and incumbent-conditioned scores divide the problem.

The work links machine learning with statistical model criticism and the logic of physical inference. The question is relevant wherever established models are applied outside the ranges in which they were built, from automated analysis of sparse experiments to audits of large scientific data collections. Code and data are available at https://github.com/ChenxiHeCam/Epistemic-probe-of-Classical-Horizons and https://doi.org/10.5281/zenodo.22138621; the exact submission release will be preserved as a new archive version.

The manuscript is not under consideration elsewhere, and all authors have approved its submission. The authors declare no competing interests.

Sincerely,

Chenxi He  
Cavendish Laboratory, University of Cambridge  
ch2067@cam.ac.uk
