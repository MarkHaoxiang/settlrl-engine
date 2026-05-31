#!/usr/bin/env bash
# Run the environment throughput benchmarks (catan-engine).
# Pass extra args to pytest-benchmark: ./run_benchmarks.sh -k batched
# Save / compare a baseline: ./run_benchmarks.sh --benchmark-save=baseline
#                            ./run_benchmarks.sh --benchmark-compare
#
# `--benchmark-only` runs just the benchmarks; `--no-cov` disables coverage
# (on by default via addopts), which otherwise instruments the hot loop and
# distorts the timings. See packages/catan-engine/tests/benchmark/README.md.
#
# Run with `uv run --package`, which builds only catan-engine (and its deps) in
# the shared workspace venv rather than syncing the whole workspace.
set -euo pipefail

uv run --package catan-engine pytest packages/catan-engine/tests/benchmark \
    --benchmark-only --no-cov "$@"
