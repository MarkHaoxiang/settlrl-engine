"""The full AlphaZero training-run state, saved/loaded for bit-exact resume.

``TrainState`` bundles everything that changes during training — the net, the
optimiser moments, the replay buffer, the iteration reached, and the best arena
win rate — so a crashed or paused run continues *identically* (the per-iteration
RNG derives from the static seed and the iteration index, both recoverable). The
optimiser and buffer *objects* are static, rebuilt from hyperparameters; only
their state lives here. Serialised with orbax, which handles the optimiser /
buffer pytrees a plain ``.npz`` cannot.

A training-side module: not imported by the package root.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, Float, Int

from settlrl_learn.model import AZParams


class TrainState(NamedTuple):
    """One AlphaZero run's complete mutable state."""

    params: AZParams
    opt_state: optax.OptState
    buffer_state: Any  # flashbax buffer state pytree
    iteration: Int[Array, ""]  # iterations completed
    best: Float[Array, ""]  # best arena win rate so far


_CKPTR = ocp.StandardCheckpointer()


def save_train_state(path: str | Path, state: TrainState) -> None:
    """Write ``state`` to ``path`` (a directory; replaces any existing one)."""
    p = Path(path).absolute()
    if p.exists():
        shutil.rmtree(p)
    _CKPTR.save(p, state)
    _CKPTR.wait_until_finished()


def load_train_state(path: str | Path, template: TrainState) -> TrainState:
    """Restore a :func:`save_train_state` checkpoint into ``template``'s
    structure — build the template by rebuilding the optimiser/buffer from the
    same hyperparameters (their shapes, not values, are what's needed)."""
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, template)
    return cast(TrainState, _CKPTR.restore(Path(path).absolute(), abstract))
