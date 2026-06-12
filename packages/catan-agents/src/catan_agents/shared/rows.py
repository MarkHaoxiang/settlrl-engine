"""The flat action table, decoded once for every agent.

Device columns drive the vmapped sweeps (greedy's tier scores, lookahead's
successor sweep, mcts transitions, the planner's tactic); the host mirrors
drive the planner's numpy-side logic. One decode here instead of one per
module keeps the row vocabulary identical everywhere.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from catan_engine.env import N_FLAT, ActionType, flat_to_action

ROW_TYPE, ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))
"""Device decode: each flat row's action type and ``(idx, target)`` params."""

ROW_IDX: np.ndarray = np.asarray(ROW_PARAMS.idx)
"""Primary action parameter per flat row (host-side decode)."""

ROW_TARGET: np.ndarray = np.asarray(ROW_PARAMS.target)
"""Secondary action parameter per flat row (host-side decode)."""

_HOST_TYPE: np.ndarray = np.asarray(ROW_TYPE)

ROWS_OF_TYPE: dict[int, np.ndarray] = {
    int(t): np.flatnonzero(t == _HOST_TYPE) for t in ActionType
}
"""Flat rows of each action type, in table order."""

_ROW_LOOKUP: dict[tuple[int, int, int], int] = {
    (int(t), int(i), int(tg)): row
    for row, (t, i, tg) in enumerate(zip(_HOST_TYPE, ROW_IDX, ROW_TARGET, strict=True))
}


def flat_row(atype: ActionType, idx: int = 0, target: int = 0) -> int:
    """The flat row of one concrete ``(action type, params)`` move."""
    return _ROW_LOOKUP[(int(atype), idx, target)]
