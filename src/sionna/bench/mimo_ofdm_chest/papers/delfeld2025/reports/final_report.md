# Final Review Report: Delfeld et al., 2025

## Final Verdict

- Verdict: `accepted with caveats`
- Faithful enough for benchmarking? Yes for the model-based RKHS estimator under the current Sionna comb-pilot benchmark.
- Main caveats:
  - The implementation is a Sionna uplink adaptation of the paper's single-user RKHS estimator.
  - RKHS normalization, sparse regularization scale, debias scale, and detector-facing error variance are benchmark-calibrated implementation choices.
  - The active runtime uses `19+15` sparse/debias iterations rather than the paper's `30+30`; these values were selected by Optuna tuning on the benchmark channel model.
  - The data-driven RKHS variant is not implemented.

## Findings

### Finding: Active estimator performs the paper's full-frequency RKHS evaluation

- Severity: low
- Description: `RKHSChannelEstimator` inherits from Sionna's full-grid
  `BaseChannelEstimator` and private `_PilotGroupMixin`, overrides `call()`,
  runs the paper's sparse RKHS/debiasing stages on pilot-comb observations,
  and evaluates the recovered channel function over all effective subcarrier
  frequencies using `S(F)` from Algorithm 1 line 15.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py`
- Recommended disposition: Accept. This is closer to Algorithm 1 line 15 than the earlier pilot-domain first pass.

### Finding: OFDM-symbol completion remains a Sionna benchmark adaptation

- Severity: low
- Description: The paper assumes a static environment and has no OFDM-symbol axis. The implementation applies the paper estimator independently to pilot-bearing OFDM symbols. Canonical `time_avg=True` averages those estimates and repeats the mean over all OFDM symbols; `time_avg=False` performs linear interpolation and extrapolation instead.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py`, `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/run_benchmark.py`
- Recommended disposition: Accept and keep documented.

### Finding: Regularization and normalization are calibrated

- Severity: medium
- Description: The implementation uses the paper's `sqrt(N0)` sparse-regularization dependence, but with benchmark-tuned and canonically rounded `lambda_scale=58.3` rather than the paper's `500.0`. It also uses unitary centered FFTs, omits an explicit `1/N` kernel prefactor, and applies the rounded benchmark-tuned `debias_scale=0.00115`.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py`, `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/run_benchmark.py`, `reports/implementation_report.md`
- Recommended disposition: Accept with caveats; these choices are necessary for a numerically useful Sionna implementation but are not fully pinned down by the paper.

### Finding: Canonical hyperparameters were selected by Optuna

- Severity: low
- Description: The calibrated Sionna 2.0.1 v2 50-trial weighted-NMSE study
  used the declared five-SNR objective and 60 SNR/channel pairs per trial,
  without incumbent injection. Its raw winner did not reproduce on paired
  canonical-size validation, which retained the current rounded parameters.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/tuning.py`,
  `src/sionna_bench/mimo_ofdm_chest/benchmarking/hyperparameter_tuning.py`, and
  the retained external v2 study evidence.
- Recommended disposition: Accept as a benchmark calibration with an explicit
  independent-validation gate.

### Finding: Corrected canonical results reproduce the preserved reference

- Severity: low
- Description: The canonical benchmark was rerun through calibrated Sionna
  with explicit UMi v19.2. Every NMSE, BER, BLER, and confidence-interval
  array exactly matches the comparison worktree output.
- Evidence: `outputs/mimo_ofdm_chest/delfeld2025/3gpp-umi/RKHSChannelEstimator.npz` and
  the retained external v2 benchmark evidence.
- Recommended disposition: Accept.

### Finding: Paper module ownership follows the shared architecture

- Severity: low
- Description: Online RKHS logic remains in `estimator.py`; `tuning.py`
  exposes `run_tuning()` and delegates to the shared Optuna helper; and
  `run_benchmark.py` exposes `run_benchmark()` and delegates to the shared
  evaluation path. No persistent asset module is needed.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py`,
  `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/tuning.py`, and
  `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/run_benchmark.py`.
- Recommended disposition: Accept.

### Finding: Detector-facing error variance is approximate

- Severity: medium
- Description: The paper does not define the channel-estimation error
  variances expected by Sionna's detector. The implementation uses
  `err_var_scale*no` and time-interpolates it. A v2 10-trial 10 dB BLER study
  selected raw `0.005575293412686262`, but rounded `0.00558` tied incumbent
  `0.00102` on paired canonical-size validation. The strict tie rule retained
  the incumbent.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py`,
  `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/tuning.py`, and
  the retained external v2 calibration evidence.
- Recommended disposition: Accept as a disclosed benchmark calibration, not
  as a paper-derived posterior variance.

### Finding: Online shapes are derived from active geometry

- Severity: low
- Description: The estimator contains no canonical grid, UE, pilot-symbol, or
  receive-antenna constants. Active nonzero pilot groups determine frequency
  and OFDM-symbol dimensions; array dimensions come from `num_rows` and
  `num_cols`; and rank is clipped independently to each available dimension.
  One-pilot groups
  and nonuniform combs are rejected before optimization. Unequal active
  pilot-symbol counts and runtime receive arrays that do not match
  `num_rows*num_cols` also fail with specific messages.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py` and
  `test/mimo_ofdm_chest/papers/delfeld2025/test_estimator.py`.
- Recommended disposition: Accept.

### Finding: Complete online inference is Tier A compilable

- Severity: low
- Description: Pilot groups are processed as a tensor dimension through
  `S`, `S^H`, sparse optimization, debiasing, and `S(F)`. Registered flat
  output indices replace inference-time tensor-to-Python coordinate
  extraction. The complete estimator call compiles as one full graph with no
  graph breaks and agrees with eager inference.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/delfeld2025/estimator.py` and
  `test/mimo_ofdm_chest/papers/delfeld2025/test_estimator.py`.
- Recommended disposition: Accept as Tier A.

## Method Audit

### Method: `RKHSChannelEstimator`

- Paper method faithfully implemented? Mostly, for the model-based RKHS estimator. The pilot-domain RKHS optimization, support extraction, debiasing, and full-frequency evaluation are represented.
- Approved plan followed? The original reviewed plan was superseded by a human-requested full-grid update.
- Surrogate-rule compliance: No alternate sparse estimator was substituted.
- Interface adaptations documented? Yes.
- Smoke/sanity checks acceptable? Yes. The current canonical result is finite;
  direct tests confirm that variance calibration leaves `h_hat` unchanged.
- Analytical evidence: direct tests cover the `S`/`S^H` adjoint identity,
  exact group soft-thresholding, fixed-support debiasing, and pilot consistency
  of low-rank full-subcarrier RKHS reconstruction.
- Remaining risk: Medium, mainly due to normalization and calibrated detector variance.

### Method: Data-driven RKHS estimator

- Paper method faithfully implemented? No.
- Reason: Learned iteration-dependent parameters, training data, checkpoint format, and validation pipeline are unavailable.
- Remaining risk: Future work if training assets are generated.

## Divergence Ledger

### Divergence: Full-grid estimator with time interpolation

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: The benchmark needs all OFDM symbols. The estimator performs the paper-defined `S(F)` function evaluation on each pilot-bearing OFDM symbol and then uses a separate benchmark-only OFDM-symbol completion stage.
- Impact on faithfulness: Low to moderate.
- Impact on benchmarking fairness: Low.

### Divergence: Unit-normalized FFT/kernel scaling

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: The paper's equations require concrete normalization choices in code. The current choices produce stable, finite RKHS estimates under the Sionna benchmark.
- Impact on faithfulness: Moderate.
- Impact on benchmarking fairness: Low to medium.

### Divergence: Tuned reduced iteration counts

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: The paper uses 30 sparse and 30 debiasing iterations; the active benchmark uses 19 and 15 for runtime and because this setting was selected by the documented Optuna calibration.
- Impact on faithfulness: Moderate.
- Impact on benchmarking fairness: Medium.

### Divergence: Benchmark-tuned regularization scales

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: The paper's simulation constants do not transfer directly to the repository's Sionna normalization. The active values were selected by a documented weighted-NMSE tuning run rather than by final leaderboard hand tuning.
- Impact on faithfulness: Moderate.
- Impact on benchmarking fairness: Medium.

## Suitability Assessment

- Strengths of the paper for this task: Clear RKHS optimization, explicit low-rank factorization, explicit FBS/debiasing algorithm, and a natural full-frequency evaluation step.
- Weaknesses of the paper for this task: Normalization choices and detector-facing variances are not fully specified for Sionna's benchmark harness.
- Required assets for omitted methods: data-driven RKHS training data, training code, learned weights, checkpoint validation, and inference loader.
