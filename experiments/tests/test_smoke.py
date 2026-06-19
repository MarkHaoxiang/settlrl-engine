"""End-to-end smoke tests: each framework runs at trivial budgets.

These exercise the whole plumbing of every experiment framework — config
resolution, the data/optimisation path, the bench gate, the saved verdict —
at budgets too small to mean anything, so they catch breakage (import errors,
shape bugs, a renamed seam) without paying for a real run. They write into a
``tmp_path`` ``Run`` rather than ``runs/`` and never assert a strength claim,
only that the framework completes and records a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import load_run
from settlrl_learn.experiment import Run


def _verdict(run_dir: Path) -> str:
    result = json.loads((run_dir / "result.json").read_text())
    assert result["verdict"] in {"pass", "fail"}
    return str(result["verdict"])


def test_0001_bench_smoke(tmp_path: Path) -> None:
    run = load_run("0001_bench_smoke")
    cfg = run.BenchSmokeConfig.resolve({}, overrides=["games=4", "batch_size=4"])
    run.run_bench(Run(tmp_path), cfg)
    _verdict(tmp_path)


@pytest.mark.slow
def test_0002_value_fitting_smoke(tmp_path: Path) -> None:
    run = load_run("0002_linear_value_fitting")
    cfg = run.ValueFittingConfig.resolve({**run.VARIANTS["smoke"], "variant": "smoke"})
    run.run_experiment(Run(tmp_path), cfg.dump())
    _verdict(tmp_path)


@pytest.mark.slow
def test_0003_neural_board_architectures_smoke(tmp_path: Path) -> None:
    run = load_run("0003_neural_board_architectures")
    cfg = run.NeuralBoardArchitecturesConfig.resolve(run.VARIANTS["smoke"])
    run.run_experiment(Run(tmp_path), cfg)
    _verdict(tmp_path)


@pytest.mark.slow
def test_0004_alphazero_smoke(tmp_path: Path) -> None:
    run = load_run("0004_alphazero")
    cfg = run.AlphaZeroConfig.resolve(run.VARIANTS["smoke"])
    run.run_experiment(Run(tmp_path), cfg)
    _verdict(tmp_path)


@pytest.mark.slow
def test_0004_alphazero_gnn_smoke(tmp_path: Path) -> None:
    run = load_run("0004_alphazero")
    cfg = run.AlphaZeroConfig.resolve(run.VARIANTS["gnn_smoke"])
    run.run_experiment(Run(tmp_path), cfg)
    _verdict(tmp_path)


@pytest.mark.slow
def test_0005_neural_leaf_smoke(tmp_path: Path) -> None:
    run = load_run("0005_neural_leaf")
    cfg = run.NeuralLeafConfig.resolve(run.VARIANTS["smoke"])
    run.run_experiment(Run(tmp_path), cfg)
    _verdict(tmp_path)
