"""Optional action-ordering lock-out: a canonical order over a turn's main-phase
free actions, to cut search-space transpositions (after cullback/canopy).

Within the MAIN phase a free action carries a *category*; only actions whose
category is uncategorised (``0`` — e.g. ``END_TURN``) or ``>=`` the highest
category taken so far this turn stay legal. So a turn's builds/buys/trades are
forced into one canonical order, collapsing the ``k!`` orderings of ``k``
independent actions to one. The order:

    play-dev (1) -> trade (2) -> buy-dev (3) -> city (4) -> road (5) -> settlement (6)

These are transposition-safe (no reachable end-of-turn position is removed)
*except* two rare cases this single-category scheme accepts (canopy spends extra
categories on them): upgrading a **same-turn-built** settlement to a city, and
maritime-trading at a **same-turn-built** port's new rate.

Pure functions; opt-in. Not part of the engine's true legality (``flat.py`` is
untouched), so the env applies it only under ``track_ordering`` and records /
the reference oracle are unaffected. The per-turn ``category`` is one int of
state, advanced like a belief: reset on a turn change, else raised by the action.

Layered beside ``belief.py``: imported by ``env`` (and settlrl-agents' search),
never the reverse.
"""

from __future__ import annotations

import jax.numpy as jnp

from settlrl_engine.board.state import BoardState, BoolScalar, GamePhase, IntScalar
from settlrl_engine.mechanics.action import N_ACTION_TYPES, ActionType
from settlrl_engine.mechanics.flat import FLAT_ATYPE, N_FLAT, FlatMaskVec

# Category per action type: 0 = uncategorised (always allowed in MAIN, e.g.
# END_TURN, and every non-main action), 1..6 the canonical main-phase order.
_CATEGORY = [0] * N_ACTION_TYPES
for _t in (
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
):
    _CATEGORY[int(_t)] = 1
_CATEGORY[int(ActionType.MARITIME_TRADE)] = 2
_CATEGORY[int(ActionType.PROPOSE_TRADE)] = 2
_CATEGORY[int(ActionType.BUY_DEVELOPMENT_CARD)] = 3
_CATEGORY[int(ActionType.BUILD_CITY)] = 4
_CATEGORY[int(ActionType.BUILD_ROAD)] = 5
_CATEGORY[int(ActionType.BUILD_SETTLEMENT)] = 6

ORDER_CATEGORY = jnp.asarray(_CATEGORY, jnp.int32)  # indexed by ActionType
_FLAT_CATEGORY = ORDER_CATEGORY[FLAT_ATYPE]  # indexed by flat-action row

__all__ = ["ORDER_CATEGORY", "next_category", "ordering_mask"]


def next_category(
    category: IntScalar, action_type: IntScalar, turn_changed: BoolScalar
) -> IntScalar:
    """The per-turn ordering category after an action: reset to 0 on a turn
    change (``current_player`` changed), else the running max with the action's
    category (non-main actions are category 0, so they never raise it)."""
    raised = jnp.maximum(category.astype(jnp.int32), ORDER_CATEGORY[action_type])
    return jnp.where(turn_changed, jnp.int32(0), raised).astype(jnp.int32)


def ordering_mask(state: BoardState, category: IntScalar) -> FlatMaskVec:
    """The legality overlay for one game: in MAIN, a flat action is allowed iff
    its category is 0 (uncategorised) or ``>= category``; outside MAIN every
    action is allowed (the lock-out is a main-phase concept). AND this into the
    real flat legality."""
    allowed = (_FLAT_CATEGORY == 0) | (category <= _FLAT_CATEGORY)
    return jnp.where(
        state.phase == int(GamePhase.MAIN), allowed, jnp.ones(N_FLAT, jnp.bool_)
    )
