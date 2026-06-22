"""The shared action-priority prior: a static priority tier per action type.

``TIER_SCORES`` is both greedy's per-row tier score (the dominant term of its
argmax) and the search's interior-node prior (`_TIER_LOGITS` in
:mod:`settlrl_search._common`, scaled there). Tiers are spaced so no per-target
bonus (|bonus| < 50) can cross between them; types sharing a tier are never both
the argmax (disjoint phases, or strict domination). The one exception is
deliberate: a *productive* MARITIME_TRADE row carries a +150 gate (in greedy's
bonus channel, not here) that lifts it over END_TURN, so priors are unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
from settlrl_engine.env import N_ACTION_TYPES, ActionType

from settlrl_search.rows import ROW_TYPE

_TIER: dict[ActionType, float] = {
    ActionType.BUILD_CITY: 900.0,
    ActionType.BUILD_SETTLEMENT: 800.0,
    ActionType.SETUP_SETTLEMENT: 800.0,
    ActionType.BUY_DEVELOPMENT_CARD: 600.0,
    ActionType.BUILD_ROAD: 500.0,
    ActionType.PLAY_KNIGHT: 400.0,
    ActionType.PLAY_ROAD_BUILDING: 400.0,
    ActionType.PLAY_YEAR_OF_PLENTY: 400.0,
    ActionType.PLAY_MONOPOLY: 400.0,
    ActionType.ROLL_DICE: 200.0,
    ActionType.SETUP_ROAD: 200.0,
    ActionType.DISCARD: 200.0,
    ActionType.MOVE_ROBBER: 200.0,
    ActionType.REJECT_TRADE: 200.0,
    ActionType.ACCEPT_TRADE: 200.0,
    ActionType.END_TURN: 100.0,
    ActionType.MARITIME_TRADE: 0.0,
    ActionType.PROPOSE_TRADE: 0.0,
}
TIER_SCORES = jnp.asarray([_TIER[ActionType(t)] for t in range(N_ACTION_TYPES)])[
    ROW_TYPE
]
"""Per-row tier score — also the search's root-prior table (scaled there)."""
