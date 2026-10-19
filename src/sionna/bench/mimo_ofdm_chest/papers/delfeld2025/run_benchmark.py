# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runner for benchmarking the Delfeld 2025 full-grid RKHS estimator.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark
    python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sionna.bench.mimo_ofdm_chest.benchmarking import evaluate_channel_estimator
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config

from .estimator import RKHSChannelEstimator


PAPER_SLUG = "delfeld2025"


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate and record one configured RKHS estimator."""
    constructor_arguments = dict(constructor_arguments or {})
    effective_arguments = resolve_constructor_arguments(
        RKHSChannelEstimator,
        (
            "num_rows",
            "num_cols",
            "rank",
            "sparse_num_iter",
            "debias_num_iter",
            "step_size",
            "lambda_scale",
            "debias_scale",
            "err_var_scale",
            "time_avg",
        ),
        constructor_arguments,
    )
    output_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    output_path = output_dir / f"{result_name}.npz"

    result = evaluate_channel_estimator(
        RKHSChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        RKHSChannelEstimator,
        result_name=result_name,
        constructor_arguments=effective_arguments,
        smoke=smoke,
    )

    print(f"[{PAPER_SLUG}] result saved to {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    """Parse benchmark and estimator arguments."""
    parser = argparse.ArgumentParser()

    benchmark = parser.add_argument_group("benchmark configuration")
    benchmark.add_argument("--smoke", action="store_true")
    benchmark.add_argument("--output-dir", default=None)
    benchmark.add_argument("--device", default=None)
    benchmark.add_argument(
        "--precision",
        choices=("single", "double"),
        default=None,
    )
    benchmark.add_argument("--batch-size", type=int, default=None)
    benchmark.add_argument("--num-mc-batches", type=int, default=None)

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument("--rank", type=int, default=None)
    estimator.add_argument(
        "--sparse-num-iter",
        type=int,
        default=None,
    )
    estimator.add_argument(
        "--debias-num-iter",
        type=int,
        default=None,
    )
    estimator.add_argument("--step-size", type=float, default=None)
    estimator.add_argument(
        "--lambda-scale",
        type=float,
        default=None,
    )
    estimator.add_argument(
        "--debias-scale",
        type=float,
        default=None,
    )
    estimator.add_argument(
        "--err-var-scale",
        type=float,
        default=None,
    )
    estimator.add_argument(
        "--time-avg",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    return parser.parse_args()


def _system_config_from_args(args: argparse.Namespace) -> SystemConfig:
    """Select the base configuration and apply explicit CLI overrides."""
    cfg = smoke_config if args.smoke else umi_3gpp_config
    overrides = {
        name: value
        for name, value in {
            "output_dir": args.output_dir,
            "device": args.device,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "num_mc_batches": args.num_mc_batches,
        }.items()
        if value is not None
    }
    return replace(cfg, **overrides)


def main() -> None:
    """Evaluate the configured RKHS estimator from the command line."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    constructor_arguments = {
        "num_rows": cfg.bs_num_rows,
        "num_cols": cfg.bs_num_cols,
    }
    constructor_arguments.update(
        {
            name: value
            for name, value in {
                "rank": args.rank,
                "sparse_num_iter": args.sparse_num_iter,
                "debias_num_iter": args.debias_num_iter,
                "step_size": args.step_size,
                "lambda_scale": args.lambda_scale,
                "debias_scale": args.debias_scale,
                "err_var_scale": args.err_var_scale,
                "time_avg": args.time_avg,
            }.items()
            if value is not None
        }
    )
    run_benchmark(
        cfg,
        result_name=RKHSChannelEstimator.__name__,
        constructor_arguments=constructor_arguments,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
