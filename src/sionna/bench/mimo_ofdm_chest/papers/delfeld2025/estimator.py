# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full-grid RKHS estimator for Delfeld et al., 2025.

This implements the paper's low-rank RKHS denoising, forward-backward
splitting, debiasing, and RKHS function evaluation at all configured
subcarrier frequencies from Sionna's extracted pilot measurements. Completion
across OFDM symbols is handled by the repository's time-domain interpolator.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import BaseChannelEstimator, LSChannelEstimator, ResourceGrid
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin
from sionna.phy.utils import flatten_last_dims

from sionna.bench.mimo_ofdm_chest.time_domain_interpolator import TimeDomainInterpolator


class RKHSChannelEstimator(BaseChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages sparsity through RKHS regularization.

        This implementation follows Algorithm 1 proposed in:

        .. container:: bibliography-box

           .. bibliography::
              :filter: key == "delfeld2025sparse"
              :list: bullet

        .. rubric:: TL;DR

        The estimator solves a low-dimensional surrogate of an optimization problem
        derived through Reproducing Kernel Hilbert Space (RKHS) theory.
        The solver uses forward-backward splitting with two stages: a regularized stage
        that identifies the channel's support, and a debiasing stage that
        reconstructs the channel within the support using Tikhonov regularization.

        The channel estimator exploits sparsity in the beamspace and the delay domain.
        The channel is estimated independently for each pilot-bearing OFDM symbol,
        and a ``TimeDomainInterpolator`` is used to complete the channel estimate across
        OFDM symbols.

        .. rubric:: Supported System Geometry

        The estimator supports different FFT sizes, UE counts, pilot-bearing OFDM symbol
        positions, and rectangular receive antenna arrays, provided that the following hold:

        - Each pilot group is a uniform comb over subcarriers with at least two non-zero pilots.
        - All pilot groups have the same length.
        - Every stream has the same number of active pilot-bearing OFDM symbols.
        - ``num_rows * num_cols`` matches the receive-antenna dimension.

        .. rubric:: Deviations from the Paper

        - The default number of iterations and the regularization weights were selected
          using Optuna, rather than following the paper's specified values.
        - The paper does not specify detector-facing error variances, so the implementation
          returns ``err_var_scale * no``, where the default value of ``err_var_scale`` was
          selected using Optuna.

        :param resource_grid: Sionna resource grid defining the pilot pattern.
        :param num_rows: Number of rows in the base-station antenna grid.
            ``num_rows*num_cols`` must match the receive-antenna dimension.
        :param num_cols: Number of columns in the base-station antenna grid.
        :param rank: Low-rank RKHS approximation rank per frequency/row/column
            dimension. The rank is clipped independently to each available
            dimension size.
        :param sparse_num_iter: Number of sparse optimization iterations using
            gradient steps and group soft-thresholding.
        :param debias_num_iter: Number of support-constrained debiasing
            iterations after sparse support extraction.
        :param step_size: Forward-backward splitting step size.
        :param lambda_scale: Scale applied to the paper's
            ``sqrt(noise variance)`` sparse regularization weight.
        :param debias_scale: Scale applied to the debiasing Tikhonov weight.
        :param err_var_scale: Nonnegative multiplicative scale applied to the LS pilot
            error variances before interpolation.
        :param time_avg: If `True`, average estimates from all pilot-bearing
            OFDM symbols and repeat the mean over every OFDM symbol. If
            `False`, interpolate linearly across OFDM symbols and extrapolate
            outside the first and last pilot-bearing symbols.
        :param precision: Optional Sionna precision setting.
        :param device: Optional torch device for Sionna buffers.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        *,
        num_rows: int,
        num_cols: int,
        rank: int = 4,
        sparse_num_iter: int = 19,
        debias_num_iter: int = 15,
        step_size: float = 1.57,
        lambda_scale: float = 58.3,
        debias_scale: float = 0.00115,
        err_var_scale: float = 0.00102,
        time_avg: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the full-grid RKHS channel estimator."""

        assert isinstance(num_rows, int) and num_rows > 0, (
            "`num_rows` must be a positive integer."
        )
        assert isinstance(num_cols, int) and num_cols > 0, (
            "`num_cols` must be a positive integer."
        )
        assert isinstance(rank, int) and rank > 0, (
            "`rank` must be a positive integer."
        )
        assert isinstance(sparse_num_iter, int) and sparse_num_iter >= 0, (
            "`sparse_num_iter` must be a non-negative integer."
        )
        assert isinstance(debias_num_iter, int) and debias_num_iter >= 0, (
            "`debias_num_iter` must be a non-negative integer."
        )
        assert isinstance(step_size, (int, float)) and math.isfinite(step_size), (
            "`step_size` must be finite."
        )
        assert step_size > 0.0, "`step_size` must be positive."
        assert isinstance(lambda_scale, (int, float)) and math.isfinite(lambda_scale), (
            "`lambda_scale` must be finite."
        )
        assert lambda_scale >= 0.0, "`lambda_scale` must be non-negative."
        assert isinstance(debias_scale, (int, float)) and math.isfinite(debias_scale), (
            "`debias_scale` must be finite."
        )
        assert debias_scale >= 0.0, "`debias_scale` must be non-negative."
        assert isinstance(err_var_scale, (int, float)) and math.isfinite(
            err_var_scale
        ), "`err_var_scale` must be finite."
        assert err_var_scale >= 0.0, "`err_var_scale` must be non-negative."
        assert isinstance(time_avg, bool), "`time_avg` must be boolean."

        super().__init__(
            resource_grid=resource_grid,
            precision=precision,
            device=device,
        )
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._num_rx_ant = self._num_rows * self._num_cols
        self._num_tx = int(self._pilot_pattern.num_tx)
        self._num_streams_per_tx = int(
            self._pilot_pattern.num_streams_per_tx
        )
        self._rank = rank
        self._sparse_num_iter = sparse_num_iter
        self._debias_num_iter = debias_num_iter
        self._step_size = step_size
        self._num_effective_subcarriers = int(
            resource_grid.num_effective_subcarriers
        )

        # Tuned constants for the current pilot-domain adaptation. The paper
        # uses a sqrt(noise variance) sparse penalty; the scale below accounts
        # for the unitary FFT convention and benchmark normalization.
        self._lambda_scale = lambda_scale
        self._debias_scale = debias_scale
        self._err_var_scale = err_var_scale
        self._init_pilot_groups(self._pilot_pattern, group_over=("subcarrier",))
        assert self._pilot_group_len >= 2, (
            "RKHS full-frequency evaluation requires at least two pilot "
            "subcarriers per group."
        )
        self._init_pilot_symbol_groups()
        self._time_interpolator = TimeDomainInterpolator(
            self._pilot_pattern,
            time_avg=time_avg,
        )

        num_pilot_subcarriers = self._pilot_group_len
        (
            self._subcarrier_rank,
            self._row_rank,
            self._column_rank,
        ) = tuple(
            min(self._rank, dimension_size)
            for dimension_size in (
                num_pilot_subcarriers,
                self._num_rows,
                self._num_cols,
            )
        )
        low_rank_factors = []
        # Build the paper's low-rank RKHS basis factors over pilot frequency,
        # array row, and array column coordinates.
        for dim_size, effective_rank in zip(
            (num_pilot_subcarriers, self._num_rows, self._num_cols),
            (self._subcarrier_rank, self._row_rank, self._column_rank),
        ):
            indices = torch.arange(
                dim_size, dtype=self.dtype, device=self._pilot_ind.device
            )
            kernel = torch.sinc(
                (indices[:, None] - indices[None, :]) / float(dim_size)
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
            dominant_indices = torch.argsort(eigenvalues, descending=True)[
                :effective_rank
            ]
            eigenvalues = torch.clamp(eigenvalues[dominant_indices], min=0.0)
            low_rank_factors.append(
                (
                    eigenvectors[:, dominant_indices].to(self.cdtype),
                    eigenvalues,
                )
            )
        subcarrier_eigenvectors, subcarrier_eigenvalues = low_rank_factors[0]
        row_factor = low_rank_factors[1][0] * torch.sqrt(
            low_rank_factors[1][1]
        ).to(self.cdtype).view(1, -1)
        column_factor = low_rank_factors[2][0] * torch.sqrt(
            low_rank_factors[2][1]
        ).to(self.cdtype).view(1, -1)
        subcarrier_factor = subcarrier_eigenvectors * torch.sqrt(
            subcarrier_eigenvalues
        ).to(self.cdtype).view(1, -1)

        rkhs_basis_pilot = (
            subcarrier_factor[:, None, None, :, None, None]
            * row_factor[None, :, None, None, :, None]
            * column_factor[None, None, :, None, None, :]
        ).reshape(
            num_pilot_subcarriers,
            self._num_rows,
            self._num_cols,
            self._subcarrier_rank * self._row_rank * self._column_rank,
        )
        self.register_buffer("_rkhs_basis_pilot", rkhs_basis_pilot)

        full_subcarrier_factors = self._build_full_subcarrier_factors(
            subcarrier_eigenvectors,
            subcarrier_eigenvalues,
        )
        rkhs_basis_full = (
            full_subcarrier_factors[:, :, None, None, :, None, None]
            * row_factor[None, None, :, None, None, :, None]
            * column_factor[None, None, None, :, None, None, :]
        ).reshape(
            self._pilot_group_num_groups,
            self._num_effective_subcarriers,
            self._num_rows,
            self._num_cols,
            self._subcarrier_rank * self._row_rank * self._column_rank,
        )
        self.register_buffer("_rkhs_basis_full", rkhs_basis_full)
        self.register_buffer(
            "_full_subcarrier_transform",
            self._build_full_subcarrier_transform(),
        )

        for dim_name, dim_size in (
            ("subcarrier", num_pilot_subcarriers),
            ("row", self._num_rows),
            ("column", self._num_cols),
        ):
            measurement_phase, atom_phase = self._build_center_phases(
                dim_size, self._pilot_ind.device
            )
            self.register_buffer(f"_measurement_phase_{dim_name}", measurement_phase)
            self.register_buffer(f"_atom_phase_{dim_name}", atom_phase)

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Estimate the full effective resource-grid channel.

        :input y: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size],
            `torch.complex`. Received OFDM resource grid.
        :input no: `torch.float`. Noise variance with a Sionna-compatible
            broadcast shape.

        :output h_hat: [batch, num_rx, num_rx_ant, num_tx,
            num_streams_per_tx, num_ofdm_symbols,
            num_effective_subcarriers], `torch.complex`. Channel estimates.
        :output err_var: Same shape as ``h_hat``, `torch.float`.
            Detector-facing channel-estimation error variances.
        """
        y_eff = self._remove_nulled_scs(y)
        y_eff_flat = flatten_last_dims(y_eff)

        pilot_ind = self._pilot_ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pilot_ind = pilot_ind.expand(
            y_eff_flat.shape[0],
            y_eff_flat.shape[1],
            y_eff_flat.shape[2],
            -1,
            -1,
            -1,
        )
        y_pilots = torch.gather(
            y_eff_flat.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, pilot_ind.shape[3], pilot_ind.shape[4], -1),
            -1,
            pilot_ind,
        )

        h_hat, err_var = self._estimate_pilot_symbol_subcarriers(y_pilots, no)
        return self._time_interpolator(h_hat, err_var)

    def _estimate_pilot_symbol_subcarriers(
        self,
        y_pilots: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the RKHS channel on all subcarriers of pilot symbols.

        Sionna compact pilots are first converted to LS observations. For each
        ``(tx, stream, pilot OFDM symbol)`` group, the paper's sparse RKHS
        optimization is solved on the pilot comb and the recovered function is
        then evaluated at all effective subcarrier frequencies with ``S(F)``.

        :param y_pilots: Received pilot observations gathered by Sionna.
        :param no: Noise variance with Sionna-compatible broadcast shape.

        :returns: Channel estimates and error variances with shape
            ``[batch, num_rx, num_rx_ant, num_tx, num_streams,
            num_pilot_ofdm_symbols, num_effective_subcarriers]``.
        """
        no = torch.as_tensor(no, dtype=self.dtype, device=y_pilots.device)
        h_ls, _ = LSChannelEstimator.estimate_at_pilot_locations(self, y_pilots, no)

        batch_size = h_ls.shape[0]
        num_rx = h_ls.shape[1]
        assert h_ls.shape[2] == self._num_rx_ant, (
            "`num_rows*num_cols` must match the receive-antenna dimension."
        )
        noise_variance_grid = self._expand_noise_variance_grid(
            no, batch_size, num_rx
        )
        clamped_noise_variance_grid = torch.clamp(noise_variance_grid, min=1e-12)

        h_ls_groups = self._gather_pilot_groups(h_ls)
        err_var_group = (
            self._err_var_scale
            * noise_variance_grid.squeeze(2).reshape(
                batch_size, num_rx, self._num_rx_ant
            )
        )

        # [B, num_rx, antennas, groups, pilot_sc] ->
        # [B, num_rx, groups, pilot_sc, rows, cols].
        h_ls_grid = h_ls_groups.reshape(
            batch_size,
            num_rx,
            self._num_rows,
            self._num_cols,
            self._pilot_group_num_groups,
            self._pilot_group_len,
        ).permute(0, 1, 4, 5, 2, 3).contiguous()

        coefficients = self._apply_measurement_adjoint(h_ls_grid)
        num_measurements = (
            self._pilot_group_len * self._num_rows * self._num_cols
        )
        regularization_weight = (
            self._lambda_scale
            * torch.sqrt(clamped_noise_variance_grid).unsqueeze(2).unsqueeze(-1)
            / math.sqrt(num_measurements)
        )
        coefficients = self._sparse_optimization(
            coefficients,
            h_ls_grid,
            regularization_weight,
        )
        debias_weight = (
            self._debias_scale
            * clamped_noise_variance_grid.unsqueeze(2).unsqueeze(-1)
        )
        coefficients = self._debias_coefficients(
            coefficients,
            h_ls_grid,
            debias_weight,
        )

        h_hat_groups = self._apply_full_subcarrier_operator(coefficients)
        h_hat_groups = h_hat_groups.permute(0, 1, 4, 5, 2, 3).reshape(
            batch_size,
            num_rx,
            self._num_rx_ant,
            self._pilot_group_num_groups,
            self._num_effective_subcarriers,
        )
        err_var_groups = err_var_group.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            self._pilot_group_num_groups,
            self._num_effective_subcarriers,
        )
        h_hat_pilot_symbols = self._scatter_full_subcarrier_groups(h_hat_groups)
        err_var_pilot_symbols = self._scatter_full_subcarrier_groups(
            err_var_groups
        )
        return h_hat_pilot_symbols, err_var_pilot_symbols

    def _apply_measurement_operator(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the paper's low-rank pilot measurement operator ``S``."""
        transformed = self._centered_fourier(
            coefficients,
            dims=(-4, -3, -2),
        )
        rkhs_basis_pilot = self._rkhs_basis_pilot[None, None, None]
        return (transformed * rkhs_basis_pilot).sum(dim=-1)

    def _apply_full_subcarrier_operator(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``S(F)`` for all pilot groups (Algorithm 1 line 15)."""
        transformed = self._centered_fourier_full_subcarriers(coefficients)
        rkhs_basis_full = self._rkhs_basis_full[None, None]
        return (transformed * rkhs_basis_full).sum(dim=-1)

    def _apply_measurement_adjoint(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the adjoint low-rank pilot operator ``S^H``."""
        rkhs_basis_pilot = self._rkhs_basis_pilot[None, None, None]
        lifted = residual.unsqueeze(-1) * torch.conj(rkhs_basis_pilot)
        return self._centered_fourier_adjoint(
            lifted,
            dims=(-4, -3, -2),
        )

    def _scatter_full_subcarrier_groups(
        self,
        grouped_values: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter full-subcarrier groups into compact pilot-symbol slots."""
        leading_shape = grouped_values.shape[:-2]
        flat_num_slots = (
            self._num_tx
            * self._num_streams_per_tx
            * self._num_pilot_symbol_groups_per_stream
        )
        output = torch.zeros(
            *leading_shape,
            flat_num_slots,
            self._num_effective_subcarriers,
            dtype=grouped_values.dtype,
            device=grouped_values.device,
        )
        index_shape = (1,) * len(leading_shape) + (
            self._pilot_group_num_groups,
            1,
        )
        indices = self._group_pilot_symbol_flat_index.reshape(
            index_shape
        ).expand_as(grouped_values)
        output = output.scatter(-2, indices, grouped_values)
        return output.reshape(
            *leading_shape,
            self._num_tx,
            self._num_streams_per_tx,
            self._num_pilot_symbol_groups_per_stream,
            self._num_effective_subcarriers,
        )

    def _sparse_optimization(
        self,
        coefficients: torch.Tensor,
        observations: torch.Tensor,
        regularization_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run Algorithm 1's sparse FBS and group-thresholding iterations."""
        for _ in range(self._sparse_num_iter):
            residual = self._apply_measurement_operator(coefficients) - observations
            gradient = self._apply_measurement_adjoint(residual)
            descent_step = coefficients - self._step_size * gradient
            group_norm = (
                descent_step.abs().square().sum(dim=-1, keepdim=True).sqrt()
            )
            shrinkage_factor = torch.clamp(
                1.0
                - (self._step_size * regularization_weight)
                / (group_norm + 1e-12),
                min=0.0,
            )
            coefficients = descent_step * shrinkage_factor
        return coefficients

    def _debias_coefficients(
        self,
        coefficients: torch.Tensor,
        observations: torch.Tensor,
        debias_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Optimize the Tikhonov objective on the fixed sparse support."""
        active_support = (
            coefficients.abs().square().sum(dim=-1, keepdim=True) > 0.0
        )
        for _ in range(self._debias_num_iter):
            residual = self._apply_measurement_operator(coefficients) - observations
            gradient = self._apply_measurement_adjoint(residual)
            gradient = gradient + 2.0 * debias_weight * coefficients
            coefficients = torch.where(
                active_support,
                coefficients - self._step_size * gradient,
                torch.zeros_like(coefficients),
            )
        return coefficients

    def _init_pilot_symbol_groups(self) -> None:
        """Register the per-group slot within each stream's pilot-symbol list."""
        tx_indices = self._pilot_group_constant_coord("tx").detach().cpu()
        stream_indices = self._pilot_group_constant_coord("stream").detach().cpu()
        symbol_indices = self._pilot_group_constant_coord("symbol").detach().cpu()
        slots = torch.empty(self._pilot_group_num_groups, dtype=torch.int64)
        num_slots = None
        for tx_index in range(self._num_tx):
            for stream_index in range(self._num_streams_per_tx):
                matches = torch.logical_and(
                    tx_indices == tx_index,
                    stream_indices == stream_index,
                )
                group_indices = torch.nonzero(matches, as_tuple=False).flatten()
                current_symbols = symbol_indices.index_select(0, group_indices)
                symbol_order = torch.argsort(current_symbols)
                ordered_group_indices = group_indices.index_select(0, symbol_order)
                if num_slots is None:
                    num_slots = int(ordered_group_indices.numel())
                assert int(ordered_group_indices.numel()) == num_slots, (
                    "RKHS time interpolation requires the same number of "
                    "pilot OFDM symbols for every stream."
                )
                for slot, group_index in enumerate(ordered_group_indices.tolist()):
                    slots[group_index] = slot
        assert num_slots is not None and num_slots > 0, (
            "RKHS time interpolation requires at least one pilot OFDM symbol."
        )
        self._num_pilot_symbol_groups_per_stream = num_slots
        flat_indices = (
            (tx_indices * self._num_streams_per_tx + stream_indices)
            * num_slots
            + slots
        )
        self.register_buffer(
            "_group_pilot_symbol_flat_index",
            flat_indices.to(device=self.device),
        )

    def _build_full_subcarrier_factors(
        self,
        subcarrier_eigenvectors: torch.Tensor,
        subcarrier_eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        """Build frequency cross-kernel factors for evaluating over ``F``."""
        pilot_positions = torch.arange(
            self._pilot_group_len,
            dtype=self.dtype,
            device=self.device,
        )
        target_positions = self._full_subcarrier_reduced_positions()
        cross_kernel = torch.sinc(
            (target_positions[..., None] - pilot_positions[None, None, :])
            / float(self._pilot_group_len)
        )
        inverse_sqrt_eigenvalues = torch.rsqrt(
            torch.clamp(subcarrier_eigenvalues, min=1e-12)
        ).to(self.cdtype)
        return (
            torch.matmul(cross_kernel.to(self.cdtype), subcarrier_eigenvectors)
            * inverse_sqrt_eigenvalues.view(1, 1, -1)
        )

    def _build_full_subcarrier_transform(self) -> torch.Tensor:
        """Return centered Fourier rows for all effective subcarriers."""
        source_size = self._pilot_group_len
        source_center = (source_size - 1.0) / 2.0
        source_positions = (
            torch.arange(source_size, dtype=self.dtype, device=self.device)
            - source_center
        )
        target_positions = self._full_subcarrier_reduced_positions() - source_center
        phase = (
            2.0
            * math.pi
            * target_positions[..., None]
            * source_positions[None, None, :]
            / float(source_size)
        )
        return torch.exp((1j * phase).to(self.cdtype)) / math.sqrt(source_size)

    def _full_subcarrier_reduced_positions(self) -> torch.Tensor:
        """Return all subcarriers in each pilot group's reduced comb units."""
        group_subcarrier_indices = self._pilot_group_varying_coord("subcarrier")
        subcarrier_spacings = torch.diff(group_subcarrier_indices, dim=-1)
        first_spacings = subcarrier_spacings[:, :1]
        assert bool(torch.all(subcarrier_spacings == first_spacings)), (
            "RKHS full-frequency evaluation requires uniform pilot combs."
        )
        group_spacings = first_spacings[:, 0].to(self.dtype)
        group_offsets = group_subcarrier_indices[:, 0].to(self.dtype)
        target_subcarriers = torch.arange(
            self._num_effective_subcarriers,
            dtype=self.dtype,
            device=self.device,
        )
        return (target_subcarriers[None, :] - group_offsets[:, None]) / (
            group_spacings[:, None]
        )

    def _expand_noise_variance_grid(
        self, no: torch.Tensor, batch_size: int, num_rx: int
    ) -> torch.Tensor:
        """Expand noise variance to [batch, num_rx, 1, num_rows, num_cols]."""
        if no.dim() == 0:
            noise_variance_grid = no.reshape(1, 1, 1, 1, 1)
        elif no.dim() == 1:
            assert no.shape[0] in (1, batch_size), (
                "`no` has incompatible batch dimension."
            )
            noise_variance_grid = no.reshape(no.shape[0], 1, 1, 1, 1)
        elif no.dim() == 2:
            assert no.shape[0] in (1, batch_size) and no.shape[1] in (1, num_rx), (
                "`no` has incompatible batch or receiver dimension."
            )
            noise_variance_grid = no.reshape(no.shape[0], no.shape[1], 1, 1, 1)
        elif no.dim() == 3:
            assert no.shape[0] in (1, batch_size) and no.shape[1] in (1, num_rx), (
                "`no` has incompatible batch or receiver dimension."
            )
            if no.shape[2] == 1:
                noise_variance_grid = no.reshape(no.shape[0], no.shape[1], 1, 1, 1)
            elif no.shape[2] == self._num_rx_ant:
                # Row-major antenna order: ant = row * num_cols + col.
                noise_variance_grid = no.reshape(
                    no.shape[0], no.shape[1], self._num_rows, self._num_cols
                ).unsqueeze(2)
            else:
                assert False, "`no` has incompatible receive-antenna dimension."
        else:
            assert False, "`no` must have rank 0, 1, 2, or 3."

        return noise_variance_grid.expand(
            batch_size, num_rx, 1, self._num_rows, self._num_cols
        )

    def _build_center_phases(
        self, dim_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build phase ramps for one centered Fourier dimension."""
        indices = torch.arange(dim_size, dtype=self.dtype, device=device)
        center_index = (dim_size - 1) / 2.0
        base_phase = -2.0 * math.pi * indices * center_index / dim_size
        atom_phase = torch.exp((1j * base_phase).to(self.cdtype))
        measurement_phase = torch.exp(
            (
                1j
                * (base_phase + 2.0 * math.pi * center_index**2 / dim_size)
            ).to(self.cdtype)
        )
        return measurement_phase, atom_phase

    def _center_phase_buffers(self):
        """Return centered Fourier phase buffers in subcarrier, row, column order."""
        measurement_phases = (
            self._measurement_phase_subcarrier,
            self._measurement_phase_row,
            self._measurement_phase_column,
        )
        atom_phases = (
            self._atom_phase_subcarrier,
            self._atom_phase_row,
            self._atom_phase_column,
        )
        return measurement_phases, atom_phases

    def _center_phase_shape(self, x: torch.Tensor, dim: int, phase: torch.Tensor):
        """Shape a 1D phase buffer for broadcasting along one tensor dimension."""
        shape = [1] * x.ndim
        shape[dim] = phase.shape[0]
        return shape

    def _centered_fourier(
        self, x: torch.Tensor, dims: tuple[int, int, int]
    ) -> torch.Tensor:
        """Apply the unitary centered positive-sign Fourier operator."""
        measurement_phases, atom_phases = self._center_phase_buffers()
        result = x
        for dim, phase in zip(dims, atom_phases):
            result = result * phase.view(
                self._center_phase_shape(result, dim, phase)
            )
        result = torch.fft.ifftn(result, dim=dims, norm="ortho")
        for dim, phase in zip(dims, measurement_phases):
            result = result * phase.view(
                self._center_phase_shape(result, dim, phase)
            )
        return result

    def _centered_fourier_full_subcarriers(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the centered Fourier part of ``S(F)`` to all groups."""
        transform = self._full_subcarrier_transform.to(
            dtype=x.dtype,
            device=x.device,
        )
        result = torch.einsum(
            "...gprck,gfp->...gfrck",
            x,
            transform,
        )
        return self._centered_fourier_spatial(result, dims=(-3, -2))

    def _centered_fourier_spatial(
        self,
        x: torch.Tensor,
        dims: tuple[int, int],
    ) -> torch.Tensor:
        """Apply the centered Fourier operator on row and column dimensions."""
        measurement_phases = (
            self._measurement_phase_row,
            self._measurement_phase_column,
        )
        atom_phases = (
            self._atom_phase_row,
            self._atom_phase_column,
        )
        result = x
        for dim, phase in zip(dims, atom_phases):
            result = result * phase.view(
                self._center_phase_shape(result, dim, phase)
            )
        result = torch.fft.ifftn(result, dim=dims, norm="ortho")
        for dim, phase in zip(dims, measurement_phases):
            result = result * phase.view(
                self._center_phase_shape(result, dim, phase)
            )
        return result

    def _centered_fourier_adjoint(
        self, x: torch.Tensor, dims: tuple[int, int, int]
    ) -> torch.Tensor:
        """Apply the Hermitian adjoint of `_centered_fourier`."""
        measurement_phases, atom_phases = self._center_phase_buffers()
        result = x
        for dim, phase in zip(dims, measurement_phases):
            result = result * torch.conj(phase).view(
                self._center_phase_shape(result, dim, phase)
            )
        result = torch.fft.fftn(result, dim=dims, norm="ortho")
        for dim, phase in zip(dims, atom_phases):
            result = result * torch.conj(phase).view(
                self._center_phase_shape(result, dim, phase)
            )
        return result
