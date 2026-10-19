# Implementation Plan: Delfeld et al., 2025

## Current Scope

This report records the current, human-approved full-grid implementation plan for `papers/delfeld2025`. It supersedes the earlier pilot-domain first-pass plan.

The implemented method is the model-based RKHS estimator from Section IV and Algorithm 1 of Delfeld et al., 2025. The data-driven RKHS estimator from Section V is not implemented in this pass.

## Harness Summary

The current task is channel estimation in the repository's Sionna OFDM MU-MIMO benchmark. The estimator input is:

```text
y  : [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
no : scalar or broadcastable Sionna noise variance
```

The estimator output is:

```text
h_hat   : [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, fft_size] complex
err_var : same shape as h_hat, float
```

The default UMi scenario uses four single-stream UEs, a `4x4` single-polarized BS array, 14 OFDM symbols, 624 effective subcarriers, and pilot OFDM symbols `(2, 11)`.

## Method: `RKHSChannelEstimator`

Status: `implement`

Paper locations:

- equations (2)-(3): noisy pilot-domain observation model;
- equations (57)-(68): low-rank RKHS factor construction;
- equations (83) and (88): efficient `S a` and `S^H e` products;
- equations (93)-(94): forward-backward splitting and group soft-thresholding;
- equations (95) and (99): support extraction and support-constrained debiasing;
- Algorithm 1 lines 3-15: sparse optimization, debiasing, and full-frequency evaluation.

Implementation plan:

1. Implement `RKHSChannelEstimator` as a `FullGridChannelEstimator` with `PilotGroupMixin`.
2. In `call()`, remove nulled subcarriers, gather Sionna compact pilot observations, and pass them to the RKHS estimation routine.
3. Use Sionna's LS pilot observation model to form pilot-domain channel measurements.
4. Group pilots by `(tx, stream, pilot OFDM symbol)` and pilot subcarrier.
5. Reshape each pilot group to `[pilot_subcarrier, antenna_row, antenna_col]`.
6. Build low-rank RKHS factors over pilot subcarriers, antenna rows, and antenna columns with rank `R=3`.
7. Implement `S`, `S^H`, and full-subcarrier `S(F)` products with centered unitary Fourier operators and RKHS basis factors.
8. Run the sparse FBS loop with group soft-thresholding.
9. Extract the active support and run support-constrained debiasing.
10. Evaluate the recovered coefficients on all effective subcarriers for each pilot-bearing OFDM symbol.
11. Use `TimeDomainInterpolator` to produce estimates on OFDM symbols without pilots.
12. Return calibrated detector-facing error variances using `err_var_scale*no`.

Canonical benchmark settings:

- `rank=3`;
- `sparse_num_iter=10`;
- `debias_num_iter=10`;
- `step_size=1.0`;
- `lambda_scale=60.0`;
- `debias_scale=0.01`;
- `err_var_scale=0.01`;
- `time_avg=True`.

## Adaptations

The paper's single-user method is applied independently to each Sionna transmit stream and pilot-bearing OFDM symbol.

The paper's static-channel setting does not specify OFDM-time interpolation. The implementation reconstructs all subcarriers natively on pilot-bearing OFDM symbols and uses `TimeDomainInterpolator` only across OFDM time.

The paper specifies `lambda=500*sqrt(N0)` in its simulation settings. The implementation keeps the `sqrt(N0)` dependence but uses `lambda_scale=60.0` under the repository's unitary FFT and Sionna benchmark normalization.

The paper does not specify Sionna detector-facing error variances. The implementation uses `err_var_scale*no` as a calibrated placeholder.

## Method: Data-Driven RKHS

Status: `not implemented`

The data-driven RKHS estimator from Section V and Algorithm 2 requires learned iteration-dependent parameters, a training dataset, training code, checkpoint format, trained weights, and validation metrics. These assets are not available in this repository.

## Verification Plan

- Run focused smoke checks for finite outputs, expected shapes, and nonnegative error variances.
- Run the canonical UMi benchmark with `python -m sionna_bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark`.
- Save the canonical result as `outputs/mimo_ofdm_chest/delfeld2025/<config-slug>/RKHSChannelEstimator.npz`.
- Document all deviations and benchmark metrics in `reports/implementation_report.md`.
