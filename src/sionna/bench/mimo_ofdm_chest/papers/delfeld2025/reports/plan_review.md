# Plan Review: Delfeld et al., 2025

## Verdict

- Reviewed plan: current full-grid `RKHSChannelEstimator` plan in `reports/plan.md`.
- Verdict: accepted with caveats.
- Methods cleared for implementation: `RKHSChannelEstimator`.
- Methods not implemented: data-driven RKHS estimator from Section V.

## Review Findings

### Full-Grid RKHS Reconstruction

The plan now implements Algorithm 1 line 15 by evaluating the recovered RKHS coefficients on all effective subcarriers for pilot-bearing OFDM symbols. This is closer to the paper than the earlier pilot-domain first pass that delegated all off-pilot reconstruction to Sionna interpolation.

Accepted caveat: OFDM-time reconstruction is still a benchmark adaptation because the paper does not define time interpolation across multiple OFDM symbols.

### Sionna Interface

Using `FullGridChannelEstimator` is appropriate because the estimator owns both pilot-domain RKHS denoising and frequency-domain reconstruction. `PilotGroupMixin` should be used for pilot grouping, and the implementation should not duplicate pilot-coordinate bookkeeping.

### Paper Fidelity

The core model-based method is preserved:

- low-rank RKHS factorization;
- centered Fourier implementation of `S` and `S^H`;
- sparse FBS loop;
- group soft-thresholding;
- support extraction;
- support-constrained debiasing;
- full-subcarrier evaluation from the recovered coefficients.

No surrogate sparse estimator is substituted.

### Required Documentation

The implementation report must document:

- all normalization choices for sinc kernels and centered Fourier operators;
- the retuned `lambda_scale=60.0`;
- `debias_scale=0.01`;
- the reduced `10+10` iteration count;
- the calibrated detector-facing `err_var_scale=0.01`;
- the use of `TimeDomainInterpolator` with `time_avg=True` for the canonical benchmark.

## Method Decisions

### `RKHSChannelEstimator`

- Status: accepted for implementation.
- Required benchmark output name: `RKHSChannelEstimator`.
- Required result path: `outputs/mimo_ofdm_chest/delfeld2025/<config-slug>/RKHSChannelEstimator.npz`.

### Data-Driven RKHS

- Status: not implemented.
- Reason: faithful implementation requires learned iteration-dependent parameters, training data, training code, checkpoint format, and validation metrics.

## Remaining Risks

- RKHS normalization is implementation-sensitive and only partly fixed by the paper.
- Detector-facing error variances are calibrated rather than paper-derived.
- Reduced iteration counts may affect final accuracy.
- The current implementation assumes a rectangular row-major BS antenna grid.
