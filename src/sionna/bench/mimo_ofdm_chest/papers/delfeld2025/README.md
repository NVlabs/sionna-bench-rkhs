# Implementation of the Estimators from Delfeld et al., 2025

Publication: J. Delfeld, G. Marti, and C. Dick, "Sparse MIMO-OFDM Channel Estimation via RKHS Regularization," arXiv:2511.20082v1, 2025.

## Abstract

Delfeld et al., 2025 propose sparse MIMO-OFDM channel estimation via a low-rank RKHS model. The implemented estimator denoises Sionna pilot-comb observations with the paper's model-based RKHS sparse optimization, support extraction, and debiasing procedure. It then evaluates the recovered channel function at all effective OFDM subcarrier frequencies using the paper's `S(F)` operator from Algorithm 1 line 15.

The paper assumes a static wireless environment and has no OFDM-symbol interpolation stage. For the benchmark, the paper's estimator is applied independently to each pilot-bearing OFDM symbol. `TimeDomainInterpolator` then completes the OFDM-symbol axis. The canonical `time_avg=True` setting averages the pilot-bearing-symbol estimates and repeats that mean over all OFDM symbols; `time_avg=False` instead uses linear interpolation and extrapolation across OFDM symbols. This stage is unrelated to delay-domain processing.

## Implemented Classes

`RKHSChannelEstimator` in `estimator.py` directly subclasses Sionna's
`BaseChannelEstimator` and uses its developer-facing `_PilotGroupMixin`. It
implements the full-grid `call()` contract rather than the pilot-estimation
plus interpolation contract of `BasePilotChannelEstimator`.

Main methods:

- `call()`: gathers Sionna pilot observations, performs the paper-defined RKHS estimation and full-frequency function evaluation, and then completes the OFDM-symbol axis.
- `_estimate_pilot_symbol_subcarriers()`: forms LS pilot measurements, solves the RKHS sparse optimization and debiasing problem for each `(tx, stream, pilot OFDM symbol)` group, and evaluates the recovered function over `F` with `S(F)`.
- `_apply_measurement_operator()` and `_apply_measurement_adjoint()`: implement
  the paper's low-rank `S` and `S^H` products.
- `_sparse_optimization()` and `_debias_coefficients()`: implement group-sparse
  forward-backward splitting and fixed-support Tikhonov debiasing.
- `_apply_full_subcarrier_operator()`: applies the paper's `S(F)` operator to
  evaluate the recovered low-rank RKHS function over all effective subcarrier
  frequencies.
- `_build_full_subcarrier_factors()`: builds RKHS frequency factors for all target subcarriers from the pilot-comb RKHS basis.
- `_centered_fourier()`, `_centered_fourier_full_subcarriers()`, and `_centered_fourier_adjoint()`: implement the centered Fourier products used for the paper's `S`, full-grid `S(F)`, and `S^H` operators.
- `_expand_noise_variance_grid()`: broadcasts Sionna noise variance to the antenna-grid layout used by the RKHS solver.

The implemented computation corresponds to:

- equations (2)-(3): noisy pilot-domain observation model;
- equations (41)-(45): RKHS interpolation and channel-function evaluation from
  pilot frequencies `Fpil` to all frequencies `F`;
- equations (57)-(68): low-rank RKHS factor construction;
- equations (83) and (88): efficient measurement and adjoint products;
- equations (93)-(94): forward-backward splitting and group soft-thresholding;
- equations (95) and (99): support extraction and support-constrained debiasing;
- Algorithm 1 lines 3-15: sparse optimization, debiasing, and full-frequency RKHS evaluation.

Benchmark outputs are named `RKHSChannelEstimator`.

The paper-local implementation follows the shared module contract:

- `estimator.py` contains `RKHSChannelEstimator` and online inference only;
- `tuning.py` exposes `run_tuning()` and delegates Optuna evaluation to
  `sionna.bench.mimo_ofdm_chest.benchmarking.tune_channel_estimator()`; and
- `run_benchmark.py` exposes `run_benchmark()` and delegates canonical metric
  evaluation and result writing to
  `sionna.bench.mimo_ofdm_chest.benchmarking.evaluate_channel_estimator()`.

No persistent model or statistical asset is required for this method.

## Supported System Geometry

All grid dimensions are derived from the active resource grid and pilot
groups. The estimator supports changed FFT sizes, UE counts, pilot-bearing
OFDM-symbol positions, and rectangular BS arrays under these constraints:

- each active frequency-pilot group is a uniform comb with at least two
  nonzero pilot subcarriers;
- all groups have the same length, and every stream has the same number of
  active pilot-bearing OFDM symbols;
- `num_rows*num_cols` equals the runtime receive-antenna dimension; and
- the requested `rank` is clipped independently for the pilot-frequency,
  antenna-row, and antenna-column factors. A short row dimension therefore
  does not unnecessarily reduce the frequency or column factor ranks.

Only nonzero pilot values define active OFDM-symbol groups. Masked positions
whose pilot value is zero are excluded from RKHS estimation and from the
subsequent OFDM-symbol completion stage.

## Compilation

The online estimator processes all pilot groups as one tensor dimension. The
fixed sparse-optimization and debiasing loops remain direct implementations of
the paper, while registered group-to-output indices avoid tensor-to-Python
scalar extraction. The complete estimator call passes
`torch.compile(..., backend="eager", fullgraph=True)` with eager-equivalent
outputs. `torch._dynamo.explain` reports one graph and no graph breaks, so the
inference path is classified as Tier A.

## Modifications and Omissions

The paper's single-user, static-channel estimator is applied independently to each Sionna transmit stream and pilot-bearing OFDM symbol. Averaging or linearly interpolating these estimates across OFDM symbols is a Sionna benchmark adaptation, not a step from the paper.

The active benchmark uses `19` sparse iterations and `15` debiasing iterations rather than the paper's `30+30` iteration count. These values were selected by a post-implementation Optuna tuning run and retained by the calibrated Sionna 2.0.1 v2 study documented below.

The sparse regularization follows the paper's `sqrt(N0)` dependence, but the proportionality factor is retuned to `lambda_scale=58.3` for the current Sionna benchmark normalization.

The debiasing Tikhonov weight is scaled by `debias_scale=0.00115`.

The paper does not specify detector-facing Sionna error variances. This implementation reports a calibrated scalar variance `err_var_scale*no` on the full subcarrier estimates at pilot-bearing OFDM symbols, with benchmark setting `err_var_scale=0.00102`; the variance is then time-interpolated together with the channel estimate.

The low-rank sinc kernels and centered Fourier operators use unitary FFT conventions and omit an explicit `1/N` prefactor that caused the operator to collapse numerically in this implementation.

The receive antenna array is assumed to be a row-major rectangular grid with `num_rows*num_cols` equal to Sionna's receive-antenna dimension.

The data-driven RKHS variant is not implemented because the learned iteration-dependent parameters, training data, checkpoint format, and validation pipeline are not available in this repository.

## Hyperparameter Tuning

The calibrated Sionna 2.0.1 v2 study used source commit
`f16208347246d8d17ec69628dfb856b19287b80c` and the explicit UMi TR 38.901
v19.2 channel model. It minimized weighted NMSE over Eb/N0 points `-10`,
`-5`, `0`, `5`, and `10` dB with linear-SNR weights
`10**(Eb/N0/10)`. Each of 50 TPE trials used three batches of four channels per
SNR. The sampler and probe seeds were `123` and `4242`. The search
ranges were rank `[1, 4]`, sparse iterations `[5, 20]`, debias iterations
`[0, 20]`, step size `[0.05, 2]`, lambda scale `[1, 300]`, and debias scale
`[1e-4, 1]`, with the three continuous ranges sampled logarithmically. The
study ran 50 requested trials: 50 completed, zero failed, and zero were
pruned.

Without historical trial injection, Optuna selected raw rank `3`, 11 sparse
iterations, 18 debias iterations, step size `0.5908971885314113`, lambda scale
`70.00947571361621`, and debias scale `0.0001562869026638533`, with objective
`0.0010243452053555885`. Paired validation used independent seed `314159` and
63 batches of four channels per SNR. The rounded candidate scored
`0.0010972862460708961`, while the current configuration scored
`0.000913108160566362`; the current parameters were therefore retained. Search
and validation took 35.22 and 45.51 seconds on an NVIDIA L40S.

The complete rounded candidate was `rank=3`, `sparse_num_iter=11`,
`debias_num_iter=18`, `step_size=0.591`, `lambda_scale=70.0`, and
`debias_scale=0.000156`.

The algorithmic study can be reproduced through the standard tuning entry
point:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
  --study rkhs \
  --output-dir /path/to/tuning/mimo_ofdm_chest/delfeld2025/rkhs_v2

python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
  --study rkhs-validation \
  --output-dir /path/to/tuning/mimo_ofdm_chest/delfeld2025/rkhs_v2
```

### Detector-Facing Error Variance

`err_var_scale` changes only the detector-facing variance and cannot change
`h_hat` or NMSE. After freezing the RKHS estimator and algorithmic parameters,
it was calibrated with the v2 detector protocol: 10 Optuna TPE trials over
logarithmic `[1e-3, 1]`, minimizing codeword BLER at 10 dB. Each trial used 63
batches of four channels. The sampler and probe seeds were `456` and `271828`.
The calibration ran 10 requested trials: 10 completed, zero failed, and zero
were pruned.

After freezing the rounded RKHS algorithm parameters, the Sionna 2.0.1 study
selected raw `0.005575293412686262`. Paired validation with independent seed
`161803` scored both the current `0.00102` and rounded `0.00558` at
`0.01488095238095238`. An exact tie retains the current value, so the canonical
scale remains `0.00102`. Search and validation took 27.83 and 14.80 seconds.

The calibration can be reproduced with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
  --study variance \
  --algorithm-output-dir /path/to/tuning/mimo_ofdm_chest/delfeld2025/rkhs_v2 \
  --output-dir /path/to/tuning/mimo_ofdm_chest/delfeld2025/variance_scale_v2

python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
  --study variance-validation \
  --algorithm-output-dir /path/to/tuning/mimo_ofdm_chest/delfeld2025/rkhs_v2 \
  --output-dir /path/to/tuning/mimo_ofdm_chest/delfeld2025/variance_scale_v2
```

The retained preflight records Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc` and tracked-diff SHA-256
`c23767c9b828f7ef7fee30ad31f1daed942c552c4e350732524dde9bc71185e6`.
The environment was Python 3.12.3, PyTorch `2.11.0+cu126`, CUDA 12.6,
driver `580.126.09`, NumPy 2.4.4, SciPy 1.17.1, and Optuna 4.9.0 on an
NVIDIA L40S and AMD EPYC 7R13. The retained `pip-freeze.txt` has SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Canonical evaluation used seed `2026`.

## Benchmark Run

Canonical benchmark command:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark
```

The currently saved canonical output was regenerated with the Sionna 2.0.1
source checkout containing the full-grid `BaseChannelEstimator` contract.
The post-RNG-correction rerun took 138.69 seconds on an NVIDIA L40S and exactly
reproduced every NMSE, BER, and BLER sample in the preserved pre-fix Sionna
2.0.1 output. Relative to the older Git regression reference, the maximum
pointwise changes are `0.102%` for NMSE, `1.83%` for BER, and `2.27%` for
BLER; these small detector-metric differences are consistent with the shared
evaluation seed and metric transition.

The default run uses `configs.umi_3gpp_config`, i.e. the Eb/N0 grid from `-10` dB to `15` dB in `2.5` dB steps, `batch_size=4`, `num_mc_batches=63`, four UEs, a `4x4` BS antenna array, 52 PRBs, and pilot-bearing OFDM symbols `(2, 11)`.

The runner evaluates `RKHSChannelEstimator` with:

- `num_rows=cfg.bs_num_rows=4`;
- `num_cols=cfg.bs_num_cols=4`;
- `rank=4`;
- `sparse_num_iter=19`;
- `debias_num_iter=15`;
- `step_size=1.57`;
- `lambda_scale=58.3`;
- `debias_scale=0.00115`;
- `err_var_scale=0.00102`;
- `time_avg=True`;
- `precision=cfg.precision`.

Results are written to:

```text
outputs/mimo_ofdm_chest/delfeld2025/3gpp-umi/RKHSChannelEstimator.npz
```

Smoke command:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark --smoke
```
