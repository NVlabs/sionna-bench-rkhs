# Implementation Report: Delfeld et al., 2025

## Approved Plan

The original reviewed plan implemented a pilot-domain first pass. After later discussion about relaxing the pilot-estimation/interpolation split, the human requested moving the model-based RKHS estimator to the repository's full-grid channel-estimator interface. This report describes the current implementation.

The implemented method is `RKHSChannelEstimator`, a full-grid adaptation of the paper's model-based RKHS estimator from Section IV and Algorithm 1. It implements sparse RKHS optimization and debiasing on Sionna pilot-comb observations, then evaluates the recovered channel function at all effective subcarrier frequencies using `S(F)` from Algorithm 1 line 15. The paper assumes a static channel; completion across the benchmark's OFDM-symbol axis is delegated to `TimeDomainInterpolator`.

## Implemented Files

- `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/tuning.py`

`estimator.py` contains only estimator construction and online inference.
`tuning.py` exposes the standard `run_tuning()` handle and delegates to
`evaluation.tune_channel_estimator()`. `run_benchmark.py` exposes the standard
`run_benchmark()` handle and delegates evaluation and persistence to
`evaluation.evaluate_channel_estimator()`. Delfeld requires no persistent
paper-local asset module.

## Implementation Notes

`RKHSChannelEstimator` inherits from Sionna's full-grid
`BaseChannelEstimator` and private `_PilotGroupMixin`.

The estimator:

- gathers Sionna pilot observations from `y`;
- computes LS pilot measurements using Sionna's LS pilot observation model;
- groups pilots by `(tx, stream, pilot OFDM symbol)` and pilot subcarrier;
- reshapes each group to the paper's subcarrier-row-column grid;
- builds low-rank RKHS factors over pilot subcarriers, antenna rows, and antenna columns;
- solves Algorithm 1's sparse forward-backward splitting loop with group soft-thresholding;
- extracts support and runs support-constrained debiasing;
- evaluates the recovered RKHS function over `F` using the paper's full-frequency `S(F)` factors;
- independently repeats that procedure for each pilot-bearing OFDM symbol; and
- completes the OFDM-symbol axis with `TimeDomainInterpolator`.

Paper equations/statements implemented:

- equations (2)-(3): pilot-domain observation model;
- equations (41)-(45): frequency cross-kernels and RKHS channel-function
  evaluation from `Fpil` to all `F`;
- equations (57)-(68): low-rank RKHS factor construction;
- equations (83) and (88): efficient `S a` and `S^H e` products using centered unitary FFTs;
- equations (93)-(94): FBS descent and group soft-thresholding;
- equations (95) and (99): support extraction and support-constrained debiasing;
- Algorithm 1 lines 3-15: sparse optimization, debiasing, and full-frequency RKHS evaluation.

The terminology was rechecked against Section IV-B, Section IV-D, and
Algorithm 1. The paper calls the mapping from `Fpil` to `F` interpolation and
function evaluation; in the reduced formulation this is the `S(F)` operation
returned by Algorithm 1 line 15. This paper-defined frequency evaluation is
distinct from `TimeDomainInterpolator`. The latter only supplies the
benchmark's otherwise absent OFDM-symbol dimension and does not perform
delay-domain interpolation.

The canonical benchmark settings are:

- `rank=4`;
- `sparse_num_iter=19`;
- `debias_num_iter=15`;
- `step_size=1.57`;
- `lambda_scale=58.3`;
- `debias_scale=0.00115`;
- `err_var_scale=0.00102`;
- `time_avg=True`.

These values were retained by the calibrated Sionna 2.0.1 v2 Optuna study. Its
50 trials used weighted NMSE over `[-10, -5, 0, 5, 10]` dB, three batches of
four channels per SNR, sampler seed `123`, and probe seed `4242`, without
historical trial injection. The raw winner had rank `3`, 11 sparse iterations,
18 debias iterations, step size `0.5908971885314113`, lambda scale
`70.00947571361621`, and debias scale `0.0001562869026638533`. On paired
seed-`314159` canonical-size validation, the rounded candidate scored
`0.0010972862460708961` and the current configuration scored
`0.000913108160566362`. The current rounded parameters therefore remained
selected. Search and validation took 35.22 and 45.51 seconds on an NVIDIA L40S.

## Deviations From The Paper

- The paper's single-user estimator is applied independently to each Sionna transmit stream and pilot-bearing OFDM symbol.
- The paper assumes a static environment and does not define interpolation
  across OFDM symbols. The canonical `time_avg=True` benchmark adaptation
  averages the independently estimated pilot-bearing symbols and repeats the
  mean across all OFDM symbols. With `time_avg=False`, the implementation uses
  linear OFDM-symbol interpolation and extrapolation instead.
- The active benchmark uses `19+15` sparse/debias iterations rather than the paper's `30+30`.
- The sparse regularization follows the paper's `sqrt(N0)` dependence, but the proportionality factor is retuned from the paper's `500.0` to `58.3` for this Sionna benchmark normalization.
- The debiasing Tikhonov weight is scaled by `debias_scale=0.00115`.
- The paper does not define detector-facing Sionna error variances. The implementation uses the calibrated placeholder `err_var_scale*no` on the full-subcarrier estimates and time-interpolates it.
- Indexless sinc kernels are normalized without an explicit `1/N` prefactor, which caused near-zero outputs under this implementation's unitary FFT convention.
- The BS antenna array is treated as a row-major rectangular grid.

## Not Implemented

The data-driven RKHS estimator from Section V and Algorithm 2 is not implemented. It requires learned iteration-dependent parameters, a training data pipeline, checkpoint format, trained weights, and validation metrics.

## Verification

Commands run during the current full-grid update:

```bash
python -m pytest test/mimo_ofdm_chest/infrastructure/test_time_domain_interpolator.py -q
python -m pytest -q test/mimo_ofdm_chest/papers/delfeld2025/test_estimator.py
python -m pytest -q
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py
```

The permanent RKHS integration test exercises complete estimation with
`rank=1`, two sparse iterations, and one debiasing iteration on three supported
systems: comb spacing/grid/UE/BS-antenna combinations `(1, 48, 1, 2)`,
`(3, 72, 3, 4)`, and `(4, 96, 4, 12)`. The shared probe validates output shape,
dtype, device, finiteness, and nonnegative error variance and requires NMSE
better than the zero estimate.

Operator-level tests additionally verify the paper-specific numerical
invariants. The implemented `S` and `S^H` satisfy the complex inner-product
adjoint identity. A zero-residual sparse step exactly matches group
soft-thresholding and removes a below-threshold group. One debiasing step lowers
the support-constrained Tikhonov objective without reactivating a zero group.
Finally, rank-two full-subcarrier RKHS evaluation reproduces the pilot operator
at every pilot subcarrier. The production operator and iteration formulas were
extracted into private methods for these tests without changing their tensor
operations.

The shape audit found no embedded canonical grid, UE, pilot-symbol, or receive-
antenna dimensions. Pilot-group length and count come from `_PilotGroupMixin`;
full-frequency size comes from the resource grid; receive-array reshaping uses
`num_rows` and `num_cols`; measurement normalization uses the runtime pilot and
array dimensions; and noise broadcasting uses runtime batch, receiver, and
antenna sizes. The requested rank is clipped independently to the active
frequency, row, and column dimensions, so a short dimension does not reduce
the other factor ranks.

Two implicit geometry assumptions were made explicit. Groups with fewer than
two active pilot subcarriers are now rejected before comb-spacing inference.
`TimeDomainInterpolator` now receives the original pilot pattern and derives
each stream's ordered active pilot-bearing OFDM symbols from its nonzero pilot
values. Its compact input axis directly represents full-subcarrier estimates,
so the RKHS estimator no longer constructs an alternative full-symbol pilot
pattern. Tests cover independent per-dimension rank clipping on all three
shared geometries, the
one-pilot and nonuniform-comb rejections, unequal active pilot-symbol counts,
receive-array mismatch, an inactive masked pilot symbol, and per-antenna noise
broadcasting.

The online RKHS path now treats pilot group as a tensor dimension throughout
`S`, `S^H`, sparse optimization, debiasing, and `S(F)`. A registered flat index
scatters the resulting full-subcarrier groups into `(tx, stream, pilot-symbol
slot)` positions. This removes the inference-time group loop and its
tensor-to-Python coordinate extraction without adding an optimized special
case. Deterministic comparisons against the preceding implementation on the
same pilot patterns agree within `3.8e-7` for channel estimates and exactly for
error variances. The complete estimator passes
`torch.compile(..., backend="eager", fullgraph=True)` with eager-equivalent
outputs; `torch._dynamo.explain` reports one graph and no graph breaks. The
online implementation is therefore Tier A.

## Detector-Facing Variance Calibration

All RKHS code and algorithmic hyperparameters were frozen before calibrating
`err_var_scale`. The v2 10-trial TPE study used logarithmic `[1e-3, 1]` and
minimized codeword BLER at 10 dB using 63 batches of four channels, sampler
seed `456`, and probe seed `271828`.

After freezing the rounded RKHS algorithm parameters, the Sionna 2.0.1 study
selected raw `0.005575293412686262`. Paired seed-`161803` validation scored
both current `0.00102` and rounded `0.00558` at `0.01488095238095238`. The
strict tie rule retained `err_var_scale=0.00102`. Search and validation took
27.83 and 14.80 seconds. A direct regression test confirms that changing the
scale leaves `h_hat` identical and scales only `err_var`.

The full study evidence is retained outside the repository. Canonical outputs
were regenerated under the current configuration schema and evaluation
seeding regime after the selection.

The Sionna 2.0.1 smoke check with `time_avg=True` succeeded at 5 dB with NMSE
`1.499552e-03`, BER `5.365888e-02`, and BLER `3.750000e-01`. Its transient
result was removed after verification.

The canonical UMi benchmark was rerun with the paper-defined full-frequency
RKHS function evaluation and `time_avg=True`, then saved to:

```text
outputs/mimo_ofdm_chest/delfeld2025/3gpp-umi/RKHSChannelEstimator.npz
```

The calibrated v2 rerun took 85.76 seconds on an NVIDIA L40S and is bit-for-bit
equal to the comparison worktree output for every NMSE, BER, BLER, and
confidence-interval array. No estimator regression was found.

Its metrics over Eb/N0
`[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` dB are:

```text
NMSE: [1.350124e-02, 8.390672e-03, 5.312219e-03, 3.404818e-03,
       2.189299e-03, 1.418477e-03, 9.349897e-04, 6.288039e-04,
       4.363978e-04, 3.136713e-04, 2.367002e-04]

BER:  [1.942023e-01, 1.472277e-01, 1.011265e-01, 6.120204e-02,
       3.430276e-02, 1.691024e-02, 8.109746e-03, 3.841223e-03,
       1.533972e-03, 4.892965e-04, 9.534188e-05]

BLER: [9.239418e-01, 8.154762e-01, 6.402116e-01, 4.305556e-01,
       2.582672e-01, 1.306217e-01, 6.415344e-02, 3.306878e-02,
       1.488095e-02, 5.952381e-03, 1.984127e-03]
```

The selected RKHS algorithm was retained after the Sionna 2.0.1 study, then
its continuous parameters were rounded under the shared canonical rule. On
identical evaluation samples, rounding changes NMSE by at most `0.102%` and
BER by at most `0.758%`; BLER is identical at every SNR. Recalibrating only
`err_var_scale` cannot change NMSE. The final 15 dB NMSE is
`2.367002e-04`.

## Remaining Risks

- RKHS normalization remains an implementation-sensitive part of the paper-to-code mapping.
- The detector-facing error variance is calibrated, not derived from the RKHS posterior.
- The active iteration count is below the paper's `30+30` and was selected by
  benchmark tuning rather than by the paper.
- The data-driven RKHS variant remains unavailable without training assets.
