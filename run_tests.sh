#!/usr/bin/env bash
# Run the full test suite.
# Pass extra args to pytest: ./run_tests.sh -v, ./run_tests.sh -k test_print_board
# To accept / update expecttest snapshots: EXPECTTEST_ACCEPT=1 ./run_tests.sh
set -euo pipefail
uv run pytest packages/catan-engine/tests/ "$@"
