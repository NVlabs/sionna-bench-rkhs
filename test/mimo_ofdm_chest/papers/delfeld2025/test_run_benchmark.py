# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Delfeld benchmark result persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sionna.bench.mimo_ofdm_chest.configs import smoke_config
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.papers.delfeld2025 import run_benchmark as runner


def test_smoke_result_records_complete_rkhs_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The RKHS result records its NPZ and all behavioral arguments."""
    cfg = replace(smoke_config, output_dir=str(tmp_path), device="cpu")
    arguments = {
        "num_rows": cfg.bs_num_rows,
        "num_cols": cfg.bs_num_cols,
    }
    expected = resolve_constructor_arguments(
        runner.RKHSChannelEstimator,
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
        arguments,
    )

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([1.0])}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(runner, "evaluate_channel_estimator", evaluate_stub)
    result_name = runner.RKHSChannelEstimator.__name__
    runner.run_benchmark(
        cfg,
        result_name=result_name,
        constructor_arguments=arguments,
        smoke=True,
    )

    output_dir = tmp_path / runner.PAPER_SLUG / "smoke" / cfg.scenario_id
    assert (output_dir / "config.json").is_file()
    inventory = json.loads((output_dir / "results.json").read_text())
    assert set(inventory["results"]) == {result_name}

    entry = inventory["results"][result_name]
    assert entry["constructor_arguments"] == expected
    assert entry["assets"] == []
    assert (output_dir / entry["file"]).is_file()
