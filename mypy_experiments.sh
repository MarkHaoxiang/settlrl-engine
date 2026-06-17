#!/usr/bin/env bash
# Type-check the experiment frameworks.
#
# Each NNNN_slug/ is checked as its own unit: the script directories are
# digit-prefixed (not importable packages) and each holds a `run.py`, so a
# single mypy invocation over all of them would collide on the bare module
# name "run". One invocation per framework keeps the names distinct. The shared
# harness now lives in `settlrl_agents.experiment` (checked by the package
# mypy), so only `new.py` and the smoke tests are left here.
set -euo pipefail

uv run mypy experiments/new.py experiments/tests
for d in experiments/[0-9][0-9][0-9][0-9]_*/; do
  uv run mypy "$d"
done
