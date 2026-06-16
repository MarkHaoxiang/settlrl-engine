#!/usr/bin/env bash
# Run the full test suite (settlrl-engine + settlrl-agents + settlrl-learn + settlrl-render).
# Pass extra args to pytest: ./run_tests.sh -v, ./run_tests.sh -k test_print_board
# To accept / update expecttest snapshots: EXPECTTEST_ACCEPT=1 ./run_tests.sh
#
# Each package is run with `uv run --package`, which builds only that package (and
# its deps) in the shared workspace venv. A plain `uv run` syncs the whole
# workspace, so an unrelated member that fails to build would block the suite.
set -euo pipefail

# pytest exit code 5 = "no tests collected" (e.g. a -k filter that only matches
# the other package); treat it as success so a filtered run doesn't abort.
run() { "$@" || [ $? -eq 5 ]; }

run uv run --package settlrl-engine pytest packages/settlrl-engine/tests/ "$@"
run uv run --package settlrl-agents pytest packages/settlrl-agents/tests/ "$@"
run uv run --package settlrl-learn pytest packages/settlrl-learn/tests/ "$@"
run uv run --package settlrl-render pytest packages/settlrl-render/tests/ "$@"
