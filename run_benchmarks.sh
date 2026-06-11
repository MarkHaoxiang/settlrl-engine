#!/usr/bin/env bash
# Run the environment throughput benchmarks. The flags re-select the benchmarks
# (deselected from the default test run), run only them, and drop coverage. See
# packages/catan-engine/tests/benchmark/README.md for options and examples.
set -euo pipefail

# -n 0 turns xdist back off: pytest-benchmark needs a single process for
# stable timings (the default addopts run the regular suite with -n auto).
uv run --package catan-engine pytest packages/catan-engine/tests/benchmark \
    -m benchmark --benchmark-only --no-cov -n 0 "$@"
