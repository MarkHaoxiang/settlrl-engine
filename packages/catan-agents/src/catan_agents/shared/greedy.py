"""A scripted greedy baseline: fixed action-type priorities, pip-weighted targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from catan_engine.board.layout import N_TILES, N_VERTICES
from catan_engine.board.resources import N_RESOURCES
from catan_engine.env import (
    N_ACTION_TYPES,
    N_FLAT,
    ActionType,
    Observation,
    flat_to_action,
)

from catan_agents.shared.policy import FlatAction, FlatMask
from catan_agents.shared.value import tile_pips, vertex_pips

# Static decode of every flat row: its action type and (idx, target) params.
_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))
_ROW_IDX = _ROW_PARAMS.idx
_ROW_TARGET = _ROW_PARAMS.target

# Priority tier per action type. Tiers are spaced so no per-target bonus (max
# 15 pips) can cross between them; types sharing a tier never co-occur legally
# (they belong to disjoint phases). MARITIME_TRADE sits below END_TURN, so the
# greedy player never trades.
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
    ActionType.END_TURN: 100.0,
    ActionType.MARITIME_TRADE: 0.0,
}
_BASE = jnp.asarray([_TIER[ActionType(t)] for t in range(N_ACTION_TYPES)])[_ROW_TYPE]

# Row groups whose bonus is target-dependent.
_VERTEX_BUILD = (
    (_ROW_TYPE == ActionType.SETUP_SETTLEMENT)
    | (_ROW_TYPE == ActionType.BUILD_SETTLEMENT)
    | (_ROW_TYPE == ActionType.BUILD_CITY)
)
_ROBBER_MOVE = (_ROW_TYPE == ActionType.MOVE_ROBBER) | (
    _ROW_TYPE == ActionType.PLAY_KNIGHT
)
_DISCARD = _ROW_TYPE == ActionType.DISCARD


def greedy_policy(key: jax.Array, obs: Observation, mask: FlatMask) -> FlatAction:
    """Highest-priority legal action, ties broken uniformly at random.

    Priorities: city > settlement > dev card > road > play dev > everything
    forced (roll/setup road/discard/robber) > end turn; never trades. Within a
    tier, settlement/city targets are scored by adjacent-tile pips, the robber
    goes to the highest-pip tile (preferring a steal), and the discard gives up
    the most-held resource.
    """
    v_pips = vertex_pips(obs["tile_number"])
    t_pips = tile_pips(obs["tile_number"])
    held = obs["self_resources"].astype(jnp.float32)
    bonus = jnp.where(
        _VERTEX_BUILD,
        v_pips[jnp.clip(_ROW_IDX, 0, N_VERTICES - 1)],
        jnp.where(
            _ROBBER_MOVE,
            t_pips[jnp.clip(_ROW_IDX, 0, N_TILES - 1)] + (_ROW_TARGET >= 0),
            jnp.where(_DISCARD, held[jnp.clip(_ROW_IDX, 0, N_RESOURCES - 1)], 0.0),
        ),
    )
    noise = jax.random.uniform(key, (N_FLAT,))
    score = _BASE + bonus + noise
    return jnp.argmax(jnp.where(mask, score, -jnp.inf))
