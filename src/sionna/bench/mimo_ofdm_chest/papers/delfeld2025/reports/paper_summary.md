# Paper Summary

- Paper ID: `delfeld2025`
- Full citation: James Delfeld, Gian Marti, and Chris Dick, "Sparse MIMO-OFDM Channel Estimation via RKHS Regularization," arXiv:2511.20082v1, Nov. 2025.
- Task: Channel estimation
- Target repository interface: full-grid Sionna channel estimation via `FullGridChannelEstimator`
- Implementer: Codex
- Date: 2026-04-27

## Core Problem

The paper estimates a single-antenna UE's MIMO-OFDM channel from noisy measurements on pilot-carrying subcarriers and all BS antennas. It models the channel as band-sparse in continuous delay-beamspace, formulates a regularized RKHS estimation problem, reduces it through a representer theorem and low-rank kernel approximations, and solves the resulting group-sparse optimization with forward-backward splitting. The paper then debiases the sparse estimate by solving a support-constrained Tikhonov problem. A data-driven deep-unfolded extension replaces the convex regularizer with a learned non-convex majorized regularizer.

## Algorithms Or Variants In Scope

- `RKHSChannelEstimator`
  Model-based RKHS estimator from Section IV and Algorithm 1. It denoises Sionna pilot-comb observations with the paper's low-rank RKHS/FBS/debiasing procedure and evaluates the recovered coefficients on all effective subcarriers for pilot-bearing OFDM symbols.

- OFDM-time interpolation
  The paper assumes a static environment and does not define an OFDM-symbol
  interpolation stage. The benchmark applies the paper estimator independently
  at pilot-bearing OFDM symbols. `TimeDomainInterpolator` then either averages
  and repeats those estimates (`time_avg=True`, canonical) or interpolates
  linearly across OFDM symbols (`time_avg=False`).

- `dd-rkhs`
  Data-driven deep-unfolded estimator from Section V and Algorithm 2. This requires learned iteration-dependent parameters and a training pipeline; it is not implemented in the current pass.

## Assumptions

- Signal or system model: The paper observes `y(z_m)=h(z_m)+n(z_m)` at pilot frequencies and all BS antenna positions; see equations (2) and (3).
- Training, pilot, or input structure: Pilots are on a comb of subcarriers. The paper's simulation uses every fourth subcarrier; the benchmark uses Sionna's Kronecker pilot pattern with pilot OFDM symbols `(2, 11)` and four UE streams.
- Required side information: BS array geometry, pilot subcarrier locations, noise variance, low-rank dimension `R`, sparsity regularization `lambda`, and iteration counts/stepsizes from Section VI-B.
- Output quantity: The paper evaluates its RKHS channel function at all
  frequencies in `F` and all antenna positions using Equations (41)-(45) and
  `S(F)` in Algorithm 1 line 15. The implementation performs this evaluation
  independently at each pilot-bearing OFDM symbol and then completes the
  benchmark's OFDM-symbol axis as described above.

## Required Assets

- Datasets: None for the model-based `RKHSChannelEstimator`.
- Learned weights: Required for `dd-rkhs`; unavailable.
- Statistics or covariance files: Not required by `RKHSChannelEstimator`.
- Other persistent assets: none for the current model-based estimator.

## Potential Reimplementation Risks

- Missing details: The paper specifies Nyquist sampling and rank `R=3`, but the practical mapping from Sionna's antenna ordering and multi-user pilot pattern to paper coordinates must be documented.
- Benchmark mismatch: The paper is static in OFDM time; the benchmark requires a full time-frequency grid over 14 OFDM symbols.
- Surrogate risk: Replacing the RKHS kernel machinery with FFT sparse recovery is not allowed. The first pass must implement the paper's low-rank RKHS/FBS measurement-grid algorithm, not a different sparse estimator.
- Numerical concerns: The RKHS FBS steps may be sensitive to normalization, stepsize, and regularization scaling.
- Other: `dd-rkhs` cannot be faithfully implemented without learned parameters or a training pipeline.
