# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_cached_detector_scale_objective,
    build_cached_nmse_objective,
    tune_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic

from .estimator import RKHSChannelEstimator

__all__ = ["run_tuning"]

TUNING_EBNO_DBS = (-10.0, -5.0, 0.0, 5.0, 10.0)
TUNING_SNR_WEIGHTS = tuple(
    10.0 ** (ebno_db / 10.0) for ebno_db in TUNING_EBNO_DBS
)
DETECTOR_CALIBRATION_EBNO_DBS = (10.0,)

ALGORITHM_SAMPLER_SEED = 123
ALGORITHM_PROBE_SEED = 4242
ALGORITHM_VALIDATION_SEED = 314159
VARIANCE_SAMPLER_SEED = 456
VARIANCE_PROBE_SEED = 271828
VARIANCE_VALIDATION_SEED = 161803

RKHS_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/delfeld2025/rkhs_v2")
VARIANCE_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/delfeld2025/variance_scale_v2")


def _fixed_estimator_kwargs() -> dict[str, object]:
    """Return constructor arguments excluded from algorithm tuning."""
    return {
        "num_rows": umi_3gpp_config.bs_num_rows,
        "num_cols": umi_3gpp_config.bs_num_cols,
    }


def _suggest_rkhs_kwargs(trial, cfg, resource_grid):
    """Suggest the model-based RKHS algorithm parameters."""
    del cfg, resource_grid
    return {
        "rank": trial.suggest_int("rank", 1, 4),
        "sparse_num_iter": trial.suggest_int("sparse_num_iter", 5, 20),
        "debias_num_iter": trial.suggest_int("debias_num_iter", 0, 20),
        "step_size": trial.suggest_float("step_size", 0.05, 2.0, log=True),
        "lambda_scale": trial.suggest_float(
            "lambda_scale",
            1.0,
            300.0,
            log=True,
        ),
        "debias_scale": trial.suggest_float(
            "debias_scale",
            1e-4,
            1.0,
            log=True,
        ),
    }


def _suggest_err_var_scale(trial, cfg, resource_grid):
    """Suggest only the detector-facing variance scale."""
    del cfg, resource_grid
    return {
        "err_var_scale": trial.suggest_float(
            "err_var_scale",
            1e-3,
            1.0,
            log=True,
        )
    }


def _round_algorithm_parameters(
    parameters: dict[str, object],
) -> dict[str, object]:
    """Round continuous parameters to three significant digits."""
    return {
        "rank": int(parameters["rank"]),
        "sparse_num_iter": int(parameters["sparse_num_iter"]),
        "debias_num_iter": int(parameters["debias_num_iter"]),
        "step_size": float(f"{float(parameters['step_size']):.3g}"),
        "lambda_scale": float(f"{float(parameters['lambda_scale']):.3g}"),
        "debias_scale": float(f"{float(parameters['debias_scale']):.3g}"),
    }


def _load_best_params(output_dir: Union[str, Path]) -> dict[str, object]:
    """Load the raw Optuna winner from one study summary."""
    with (Path(output_dir) / "best_params.json").open(encoding="utf-8") as file:
        return json.load(file)["best_params"]


def _write_validation(
    output_dir: Union[str, Path],
    result: dict[str, object],
) -> dict[str, object]:
    """Write and return one paired-validation result."""
    write_json_atomic(Path(output_dir) / "paired_validation.json", result)
    return result


def _selected_algorithm_parameters(
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Return the constructor parameters selected by paired validation."""
    path = Path(algorithm_output_dir) / "paired_validation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"RKHS paired validation is required before calibration: {path}."
        )
    with path.open(encoding="utf-8") as file:
        return dict(json.load(file)["selected_parameters"])


def _run_rkhs_tuning(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
):
    """Tune RKHS algorithm parameters on one shared cached probe set."""
    objective = build_cached_nmse_objective(
        RKHSChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_PROBE_SEED,
    )
    return tune_channel_estimator(
        RKHSChannelEstimator,
        umi_3gpp_config,
        _suggest_rkhs_kwargs,
        output_dir,
        fixed_estimator_kwargs=_fixed_estimator_kwargs(),
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        n_trials=n_trials,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        sampler_seed=ALGORITHM_SAMPLER_SEED,
        probe_seed=ALGORITHM_PROBE_SEED,
        estimator_objective=objective,
    )


def _run_rkhs_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded RKHS winner with the current configuration."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            RKHSChannelEstimator,
            (
                "rank",
                "sparse_num_iter",
                "debias_num_iter",
                "step_size",
                "lambda_scale",
                "debias_scale",
            ),
        ),
        "rounded_winner": _round_algorithm_parameters(raw),
    }
    objective = build_cached_nmse_objective(
        RKHSChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_VALIDATION_SEED,
    )
    objectives = {
        name: objective(
            {
                **_fixed_estimator_kwargs(),
                **parameters,
            }
        )
        for name, parameters in candidates.items()
    }
    selected = (
        "rounded_winner"
        if objectives["rounded_winner"] < objectives["current"]
        else "current"
    )
    return _write_validation(
        output_dir,
        {
            "protocol_version": 2,
            "metric": "weighted_nmse",
            "ebno_dbs": list(TUNING_EBNO_DBS),
            "snr_weights": list(TUNING_SNR_WEIGHTS),
            "seed": ALGORITHM_VALIDATION_SEED,
            "num_mc_batches_per_snr": num_mc_batches,
            "batch_size": batch_size,
            "raw_winner": raw,
            "candidates": candidates,
            "objectives": objectives,
            "selected": selected,
            "selected_parameters": candidates[selected],
        },
    )


def _variance_objective(
    *,
    algorithm_parameters: dict[str, object],
    num_mc_batches: int,
    batch_size: int,
    seed: int,
):
    """Build the cached 10 dB codeword-BLER scale objective."""
    return build_cached_detector_scale_objective(
        RKHSChannelEstimator,
        umi_3gpp_config,
        {
            **_fixed_estimator_kwargs(),
            **algorithm_parameters,
        },
        scale_argument="err_var_scale",
        ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=seed,
        detector_metric="bler",
    )


def _run_variance_calibration(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
):
    """Calibrate the variance scale against 10 dB codeword BLER."""
    parameters = _selected_algorithm_parameters(algorithm_output_dir)
    return tune_channel_estimator(
        RKHSChannelEstimator,
        umi_3gpp_config,
        _suggest_err_var_scale,
        output_dir,
        ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
        objective_metric="detector",
        detector_metric="bler",
        n_trials=n_trials,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        sampler_seed=VARIANCE_SAMPLER_SEED,
        probe_seed=VARIANCE_PROBE_SEED,
        estimator_objective=_variance_objective(
            algorithm_parameters=parameters,
            num_mc_batches=num_mc_batches,
            batch_size=batch_size,
            seed=VARIANCE_PROBE_SEED,
        ),
    )


def _run_variance_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded scale winner with the current scale."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            RKHSChannelEstimator,
            ("err_var_scale",),
        )["err_var_scale"],
        "rounded_winner": float(f"{float(raw['err_var_scale']):.3g}"),
    }
    objective = _variance_objective(
        algorithm_parameters=_selected_algorithm_parameters(
            algorithm_output_dir
        ),
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=VARIANCE_VALIDATION_SEED,
    )
    objectives = {
        name: objective({"err_var_scale": scale})
        for name, scale in candidates.items()
    }
    selected = (
        "rounded_winner"
        if objectives["rounded_winner"] < objectives["current"]
        else "current"
    )
    return _write_validation(
        output_dir,
        {
            "protocol_version": 2,
            "metric": "codeword_bler",
            "ebno_dbs": list(DETECTOR_CALIBRATION_EBNO_DBS),
            "seed": VARIANCE_VALIDATION_SEED,
            "num_mc_batches_per_snr": num_mc_batches,
            "batch_size": batch_size,
            "raw_winner": raw,
            "candidates": candidates,
            "objectives": objectives,
            "selected": selected,
            "selected_scale": candidates[selected],
        },
    )


def run_tuning(
    *,
    study: str = "rkhs",
    n_trials: Optional[int] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    algorithm_output_dir: Optional[Union[str, Path]] = None,
):
    r"""

    .. rubric:: Algorithmic Tuning

    The ``"rkhs"`` stage minimizes weighted NMSE at -10, -5, 0, 5,
    and 10 dB. It runs 50 seeded TPE trials. Every trial evaluates three
    batches of four channels at each SNR.

    .. list-table::
       :header-rows: 1

       * - Parameter
         - Search space
       * - ``rank``
         - integer from 1 through 4
       * - ``sparse_num_iter``
         - integer from 5 through 20
       * - ``debias_num_iter``
         - integer from 0 through 20
       * - ``step_size``
         - logarithmic from 0.05 through 2
       * - ``lambda_scale``
         - logarithmic from 1 through 300
       * - ``debias_scale``
         - logarithmic from 1e-4 through 1

    The ``"rkhs-validation"`` stage compares the rounded winner with the
    existing estimator defaults using 63 batches of four channels per SNR.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` stage tunes ``err_var_scale`` logarithmically from
    1e-3 through 1 using 10 TPE trials. Codeword BLER is evaluated at
    10 dB over 63 batches of four channels. T

    The ``"variance-validation"`` stage compares the rounded winner with
    the existing scale.

    :param study: Stage to execute: ``"rkhs"``, ``"rkhs-validation"``,
        ``"variance"``, or ``"variance-validation"``.
    :param n_trials: Trial-count override. If `None`, use 50 for RKHS
        tuning and 10 for variance calibration. Validation stages do not
        run Optuna trials.
    :param num_mc_batches: Monte Carlo budget override. If `None`, use
        three batches per SNR for RKHS tuning and 63 for validation and
        variance calibration.
    :param batch_size: Channels per batch. If `None`, use the canonical
        configuration value of four.
    :param output_dir: Output directory override for the selected stage.
    :param algorithm_output_dir: Directory containing the RKHS tuning and
        validation evidence required by variance calibration.
    :returns: An Optuna study for tuning stages or a dictionary containing
        paired-validation results for validation stages.
    """
    effective_batch_size = (
        umi_3gpp_config.batch_size if batch_size is None else int(batch_size)
    )
    algorithm_output_dir = (
        RKHS_TUNING_OUTPUT_DIR
        if algorithm_output_dir is None
        else algorithm_output_dir
    )
    if study == "rkhs":
        return _run_rkhs_tuning(
            n_trials=50 if n_trials is None else n_trials,
            num_mc_batches=3 if num_mc_batches is None else num_mc_batches,
            batch_size=effective_batch_size,
            output_dir=(
                algorithm_output_dir if output_dir is None else output_dir
            ),
        )
    if study == "rkhs-validation":
        return _run_rkhs_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=(
                algorithm_output_dir if output_dir is None else output_dir
            ),
        )
    variance_output_dir = (
        VARIANCE_TUNING_OUTPUT_DIR if output_dir is None else output_dir
    )
    if study == "variance":
        return _run_variance_calibration(
            n_trials=10 if n_trials is None else n_trials,
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=variance_output_dir,
            algorithm_output_dir=algorithm_output_dir,
        )
    if study == "variance-validation":
        return _run_variance_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=variance_output_dir,
            algorithm_output_dir=algorithm_output_dir,
        )
    raise ValueError(
        "`study` must be 'rkhs', 'rkhs-validation', 'variance', or "
        "'variance-validation'."
    )


def main() -> None:
    """Parse command-line arguments and run one RKHS tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=(
            "rkhs",
            "rkhs-validation",
            "variance",
            "variance-validation",
        ),
        default="rkhs",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--num-mc-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--algorithm-output-dir", type=str, default=None)
    args = parser.parse_args()
    run_tuning(
        study=args.study,
        n_trials=args.n_trials,
        num_mc_batches=args.num_mc_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        algorithm_output_dir=args.algorithm_output_dir,
    )


if __name__ == "__main__":
    main()
