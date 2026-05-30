"""RL environment entry point.

Exposes the engine's batched transition as a single ``(action_type, params)``
interface over all 15 actions. ``step`` and ``available`` are ``jit(vmap(...))``
over the single-game dispatchers in ``action_vec``, so they run a whole batch of
games at once:

- ``action_type``: ``(batch,)`` int array of ``ActionType`` codes.
- ``params``: an ``ActionParams`` whose leaves are batched (``index`` /
  ``target`` are ``(batch,)``; ``resources`` is ``(batch, N_RESOURCES)``).

``step`` returns the new batched ``BoardState`` and a ``(batch,)`` array of
``ActionResult`` codes; illegal actions leave their game unchanged and report
``INVALID``. ``available`` returns the ``(batch,)`` legality mask without
applying anything (useful for action masking).

The ``ActionType`` / ``ActionParams`` packing convention is documented on those
types in ``catan_engine.action_vec``.
"""

from __future__ import annotations

from typing import cast

import jax

from catan_engine.action_vec import (
    ActionParams,
    ActionResult,
    ActionType,
    N_ACTION_TYPES,
    action_available,
    apply_action,
)
from catan_engine.board import Board
from catan_engine.state import BoardState

__all__ = [
    "ActionParams",
    "ActionResult",
    "ActionType",
    "N_ACTION_TYPES",
    "step",
    "available",
]

_step = jax.jit(jax.vmap(apply_action, in_axes=(0, 0, 0, 0)))
_available = jax.jit(jax.vmap(action_available, in_axes=(0, 0, 0, 0)))


def step(
    board: Board, action_type: jax.Array, params: ActionParams
) -> tuple[BoardState, jax.Array]:
    """Apply one (batched) action per game; return (new state, ActionResult codes)."""
    new_state, result = _step(board[0], board[1], action_type, params)
    return new_state, result


def available(board: Board, action_type: jax.Array, params: ActionParams) -> jax.Array:
    """``(batch,)`` legality mask for the chosen action per game (no state change)."""
    return cast(jax.Array, _available(board[0], board[1], action_type, params))
