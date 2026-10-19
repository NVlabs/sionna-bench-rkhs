# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""System-variant integration tests for the Delfeld RKHS estimator."""

from __future__ import annotations

import pytest
import torch
from sionna.phy.ofdm import PilotPattern

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import OFDMChannelEstimationSystem
from sionna.bench.mimo_ofdm_chest.papers.delfeld2025.estimator import RKHSChannelEstimator
from sionna.phy.ofdm.channel_estimation import _pilot_ind_from_pattern
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    run_nmse_probe,
    system_variant_config,
    validate_estimator_output,
)


pytestmark = pytest.mark.system_variants


def _analytical_estimator(
    system_variant: str = "baseline",
    **estimator_kwargs,
) -> RKHSChannelEstimator:
    """Build a double-precision estimator for operator-level checks."""
    cfg = system_variant_config(system_variant, precision="double")
    resource_grid = build_resource_grid(cfg)
    kwargs = {
        "rank": 1,
        "sparse_num_iter": 0,
        "debias_num_iter": 0,
        "step_size": 0.05,
        **estimator_kwargs,
    }
    return RKHSChannelEstimator(
        resource_grid,
        num_rows=cfg.bs_num_rows,
        num_cols=cfg.bs_num_cols,
        precision=cfg.precision,
        device=cfg.device,
        **kwargs,
    )


def _coefficient_shape(estimator: RKHSChannelEstimator) -> tuple[int, ...]:
    """Return one batch of RKHS coefficients accepted by ``S``."""
    return (
        1,
        1,
        estimator._pilot_group_num_groups,
        estimator._pilot_group_len,
        estimator._num_rows,
        estimator._num_cols,
        estimator._rkhs_basis_pilot.shape[-1],
    )


@pytest.mark.parametrize(
    (
        "system_variant",
        "expected_comb_spacing",
        "expected_fft_size",
        "expected_num_ue",
        "expected_num_rx_ant",
    ),
    [
        pytest.param("baseline", 1, 48, 1, 2, id="spacing1-grid48-ue1-rx2"),
        pytest.param("alternate", 3, 72, 3, 4, id="spacing3-grid72-ue3-rx4"),
        pytest.param("wide", 4, 96, 4, 12, id="spacing4-grid96-ue4-rx12"),
    ],
)
def test_rkhs_estimation_across_supported_system_variants(
    system_variant: str,
    expected_comb_spacing: int,
    expected_fft_size: int,
    expected_num_ue: int,
    expected_num_rx_ant: int,
) -> None:
    """RKHS inference supports changed comb, grid, UE, and BS dimensions."""
    cfg = system_variant_config(system_variant)
    assert cfg.fft_size == expected_fft_size
    assert cfg.num_ue == expected_num_ue
    assert cfg.num_rx_ant == expected_num_rx_ant
    resource_grid = build_resource_grid(cfg)
    estimator = RKHSChannelEstimator(
        resource_grid,
        num_rows=cfg.bs_num_rows,
        num_cols=cfg.bs_num_cols,
        rank=1,
        sparse_num_iter=2,
        debias_num_iter=1,
        step_size=0.5,
        lambda_scale=1.0,
        debias_scale=1e-3,
        precision=cfg.precision,
        device=cfg.device,
    )

    pilot_subcarrier_indices = estimator._pilot_group_varying_coord(
        "subcarrier"
    )
    subcarrier_spacings = torch.diff(pilot_subcarrier_indices, dim=-1)
    assert bool(torch.all(subcarrier_spacings == expected_comb_spacing))

    system = OFDMChannelEstimationSystem(cfg, estimator, resource_grid)
    result = run_nmse_probe(system, snr_db=5.0)

    assert result.nmse < result.zero_estimate_nmse


@pytest.mark.parametrize(
    ("system_variant", "expected_effective_ranks"),
    [
        ("baseline", (4, 1, 2)),
        ("alternate", (4, 2, 2)),
        ("wide", (4, 2, 4)),
    ],
)
def test_low_rank_basis_uses_active_geometry(
    system_variant: str,
    expected_effective_ranks: tuple[int, int, int],
) -> None:
    """Requested rank is clipped independently in each tensor dimension."""
    estimator = _analytical_estimator(system_variant, rank=4)
    expected_basis_rank = (
        expected_effective_ranks[0]
        * expected_effective_ranks[1]
        * expected_effective_ranks[2]
    )

    assert (
        estimator._subcarrier_rank,
        estimator._row_rank,
        estimator._column_rank,
    ) == expected_effective_ranks
    assert estimator._rkhs_basis_pilot.shape == (
        estimator._pilot_group_len,
        estimator._num_rows,
        estimator._num_cols,
        expected_basis_rank,
    )
    assert estimator._rkhs_basis_full.shape == (
        estimator._pilot_group_num_groups,
        estimator._num_effective_subcarriers,
        estimator._num_rows,
        estimator._num_cols,
        expected_basis_rank,
    )


def test_single_pilot_subcarrier_groups_are_rejected() -> None:
    """Full-frequency spacing inference requires two pilots in every group."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    pilots = torch.zeros_like(resource_grid.pilot_pattern.pilots)
    pilots[..., 0] = 1.0
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )

    with pytest.raises(
        AssertionError,
        match="requires at least two pilot subcarriers per group",
    ):
        RKHSChannelEstimator(
            resource_grid,
            num_rows=cfg.bs_num_rows,
            num_cols=cfg.bs_num_cols,
            rank=1,
            sparse_num_iter=0,
            debias_num_iter=0,
            precision=cfg.precision,
            device=cfg.device,
        )


def test_nonuniform_pilot_combs_are_rejected() -> None:
    """RKHS full-frequency evaluation requires uniform pilot spacing."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    pilot_pattern = resource_grid.pilot_pattern
    pilot_indices = _pilot_ind_from_pattern(pilot_pattern, device=cfg.device)
    pilot_subcarrier_indices = pilot_indices % int(
        resource_grid.num_effective_subcarriers
    )
    selected_subcarriers = torch.tensor([0, 1, 3], device=cfg.device)
    selected_pilots = torch.isin(
        pilot_subcarrier_indices,
        selected_subcarriers,
    )
    pilots = torch.where(
        selected_pilots,
        pilot_pattern.pilots,
        torch.zeros_like(pilot_pattern.pilots),
    )
    resource_grid.pilot_pattern = PilotPattern(
        pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )

    with pytest.raises(
        AssertionError,
        match="requires uniform pilot combs",
    ):
        RKHSChannelEstimator(
            resource_grid,
            num_rows=cfg.bs_num_rows,
            num_cols=cfg.bs_num_cols,
            rank=1,
            sparse_num_iter=0,
            debias_num_iter=0,
            precision=cfg.precision,
            device=cfg.device,
        )


def test_unequal_active_pilot_symbol_counts_are_rejected() -> None:
    """Every stream must provide the same compact OFDM-time axis length."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    pilot_pattern = resource_grid.pilot_pattern
    pilots = pilot_pattern.pilots.clone()
    pilot_indices = _pilot_ind_from_pattern(pilot_pattern, device=cfg.device)
    pilot_symbol_indices = pilot_indices // int(
        resource_grid.num_effective_subcarriers
    )
    deactivated_symbol = cfg.pilot_ofdm_symbol_indices[-1]
    pilots[0][pilot_symbol_indices[0] == deactivated_symbol] = 0.0
    resource_grid.pilot_pattern = PilotPattern(
        pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )

    with pytest.raises(
        AssertionError,
        match="same number of pilot OFDM symbols for every stream",
    ):
        RKHSChannelEstimator(
            resource_grid,
            num_rows=cfg.bs_num_rows,
            num_cols=cfg.bs_num_cols,
            rank=1,
            sparse_num_iter=0,
            debias_num_iter=0,
            precision=cfg.precision,
            device=cfg.device,
        )


def test_receive_array_shape_mismatch_is_rejected_before_optimization() -> None:
    """Runtime antennas must match the configured rectangular RKHS array."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    estimator = RKHSChannelEstimator(
        resource_grid,
        num_rows=1,
        num_cols=1,
        rank=1,
        sparse_num_iter=0,
        debias_num_iter=0,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.full(
        (1, 1, cfg.num_rx_ant),
        0.1,
        dtype=torch.float32,
    )

    with pytest.raises(
        AssertionError,
        match="must match the receive-antenna dimension",
    ):
        estimator(received_grid, noise_variance)


def test_ofdm_symbol_completion_uses_active_nonzero_pilot_groups() -> None:
    """Zero-valued masked pilots do not create phantom OFDM-symbol slots."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    pilots = resource_grid.pilot_pattern.pilots.clone()
    pilot_indices = _pilot_ind_from_pattern(
        resource_grid.pilot_pattern,
        device=cfg.device,
    )
    pilot_symbol_indices = pilot_indices // int(
        resource_grid.num_effective_subcarriers
    )
    deactivated_symbol = cfg.pilot_ofdm_symbol_indices[-1]
    pilots[pilot_symbol_indices == deactivated_symbol] = 0.0
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )
    estimator = RKHSChannelEstimator(
        resource_grid,
        num_rows=cfg.bs_num_rows,
        num_cols=cfg.bs_num_cols,
        rank=1,
        sparse_num_iter=1,
        debias_num_iter=0,
        lambda_scale=0.0,
        precision=cfg.precision,
        device=cfg.device,
    )

    assert estimator._num_pilot_symbol_groups_per_stream == 1
    torch.testing.assert_close(
        estimator._time_interpolator._pilot_ofdm_symbol_indices,
        torch.full(
            (cfg.num_ue, cfg.num_streams_per_tx, 1),
            cfg.pilot_ofdm_symbol_indices[0],
            dtype=torch.int64,
        ),
    )

    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.full(
        (1, 1, cfg.num_rx_ant),
        0.1,
        dtype=torch.float32,
    )
    channel_estimate, error_variance = estimator(
        received_grid,
        noise_variance,
    )
    expected_channel = torch.empty(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    validate_estimator_output(
        channel_estimate,
        error_variance,
        expected_channel,
    )


def test_measurement_operator_and_adjoint_satisfy_inner_product_identity() -> None:
    """The implemented ``S^H`` is the Hermitian adjoint of ``S``."""
    estimator = _analytical_estimator("alternate", rank=2)
    coefficients = torch.randn(
        *_coefficient_shape(estimator),
        dtype=torch.complex128,
    )
    measurements = torch.randn(
        *coefficients.shape[:-1],
        dtype=torch.complex128,
    )

    forward_inner_product = torch.vdot(
        estimator._apply_measurement_operator(coefficients).flatten(),
        measurements.flatten(),
    )
    adjoint_inner_product = torch.vdot(
        coefficients.flatten(),
        estimator._apply_measurement_adjoint(measurements).flatten(),
    )

    torch.testing.assert_close(
        forward_inner_product,
        adjoint_inner_product,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sparse_step_applies_group_soft_thresholding() -> None:
    """One zero-residual FBS step performs the expected group shrinkage."""
    step_size = 0.2
    regularization_weight = torch.tensor(0.5, dtype=torch.float64)
    estimator = _analytical_estimator(
        sparse_num_iter=1,
        step_size=step_size,
    )
    coefficients = torch.ones(
        *_coefficient_shape(estimator),
        dtype=torch.complex128,
    )
    coefficients[:, :, :, 0, 0, 0] = 0.01
    observations = estimator._apply_measurement_operator(coefficients)

    sparse_coefficients = estimator._sparse_optimization(
        coefficients,
        observations,
        regularization_weight,
    )
    group_norm = coefficients.abs().square().sum(dim=-1, keepdim=True).sqrt()
    expected = coefficients * torch.clamp(
        1.0 - step_size * regularization_weight / (group_norm + 1e-12),
        min=0.0,
    )

    torch.testing.assert_close(sparse_coefficients, expected)
    assert bool(torch.all(sparse_coefficients[:, :, :, 0, 0, 0] == 0.0))
    assert bool(torch.any(sparse_coefficients[:, :, :, 1:] != 0.0))


def test_debiasing_updates_only_the_fixed_sparse_support() -> None:
    """Debiasing lowers its objective without activating excluded groups."""
    debias_weight = torch.tensor(0.01, dtype=torch.float64)
    estimator = _analytical_estimator(
        debias_num_iter=1,
        step_size=0.01,
    )
    coefficients = torch.randn(
        *_coefficient_shape(estimator),
        dtype=torch.complex128,
    )
    coefficients[:, :, :, 0] = 0.0
    target_coefficients = 1.5 * coefficients
    observations = estimator._apply_measurement_operator(target_coefficients)

    def objective(value: torch.Tensor) -> torch.Tensor:
        residual = estimator._apply_measurement_operator(value) - observations
        return (
            0.5 * residual.abs().square().sum()
            + debias_weight * value.abs().square().sum()
        )

    objective_before = objective(coefficients)
    debiased = estimator._debias_coefficients(
        coefficients,
        observations,
        debias_weight,
    )

    assert bool(torch.all(debiased[:, :, :, 0] == 0.0))
    assert float(objective(debiased)) < float(objective_before)


def test_low_rank_full_grid_reconstruction_matches_pilot_operator() -> None:
    """Full-frequency RKHS evaluation reproduces ``S`` at pilot subcarriers."""
    estimator = _analytical_estimator("alternate", rank=2)
    assert estimator._rkhs_basis_pilot.shape[-1] == 2**3
    coefficients = torch.randn(
        *_coefficient_shape(estimator),
        dtype=torch.complex128,
    )

    pilot_measurements = estimator._apply_measurement_operator(coefficients)
    full_reconstruction = estimator._apply_full_subcarrier_operator(coefficients)
    pilot_subcarrier_indices = estimator._pilot_group_varying_coord(
        "subcarrier"
    )
    gather_indices = pilot_subcarrier_indices.reshape(
        1,
        1,
        estimator._pilot_group_num_groups,
        estimator._pilot_group_len,
        1,
        1,
    ).expand_as(pilot_measurements)
    reconstructed_pilots = torch.gather(
        full_reconstruction,
        dim=3,
        index=gather_indices,
    )

    torch.testing.assert_close(
        reconstructed_pilots,
        pilot_measurements,
        rtol=1e-10,
        atol=1e-10,
    )


def test_err_var_scale_changes_only_detector_facing_variance() -> None:
    """Variance calibration leaves the RKHS channel estimate unchanged."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    common_kwargs = {
        "num_rows": cfg.bs_num_rows,
        "num_cols": cfg.bs_num_cols,
        "rank": 1,
        "sparse_num_iter": 1,
        "debias_num_iter": 1,
        "step_size": 0.5,
        "lambda_scale": 1.0,
        "debias_scale": 1e-3,
        "precision": cfg.precision,
        "device": cfg.device,
    }
    reference_estimator = RKHSChannelEstimator(
        resource_grid,
        err_var_scale=0.01,
        **common_kwargs,
    )
    scaled_estimator = RKHSChannelEstimator(
        resource_grid,
        err_var_scale=0.025,
        **common_kwargs,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.full(
        (1, 1, cfg.num_rx_ant),
        0.1,
        dtype=torch.float32,
    )

    reference_channel, reference_variance = reference_estimator(
        received_grid,
        noise_variance,
    )
    scaled_channel, scaled_variance = scaled_estimator(
        received_grid,
        noise_variance,
    )

    torch.testing.assert_close(
        scaled_channel,
        reference_channel,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scaled_variance,
        2.5 * reference_variance,
    )


@pytest.mark.compile
def test_rkhs_estimator_compiles_as_full_graph() -> None:
    """Complete RKHS inference compiles without graph breaks."""
    torch._dynamo.reset()
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    estimator = RKHSChannelEstimator(
        resource_grid,
        num_rows=cfg.bs_num_rows,
        num_cols=cfg.bs_num_cols,
        rank=1,
        sparse_num_iter=2,
        debias_num_iter=1,
        step_size=0.5,
        lambda_scale=1.0,
        debias_scale=1e-3,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.full(
        (1, 1, cfg.num_rx_ant),
        0.1,
        dtype=torch.float32,
    )

    assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        noise_variance,
        fullgraph=True,
    )
