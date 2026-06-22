"""The shipped play path stays dependency-light: ``import settlrl_learn`` (the
plain-JAX features + MLP) must pull none of the training libraries, so a trained
model ships without them.

Run in a fresh interpreter (a subprocess) -- the test suite itself imports the
training modules, so checking ``sys.modules`` in-process would always fail."""

from __future__ import annotations

import subprocess
import sys

_TRAINING_ONLY = ("equinox", "flashbax", "optax", "orbax", "jraph")


def test_import_settlrl_learn_pulls_no_training_deps() -> None:
    code = (
        "import settlrl_learn, sys;"
        f"bad=[m for m in {_TRAINING_ONLY!r} if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = out.stdout.strip()
    assert not leaked, f"import settlrl_learn leaked training deps: {leaked}"
