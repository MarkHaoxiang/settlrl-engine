"""The scripted greedy agent: a weighting of the hand-engineered features.

The features (target build, needs/surplus, port ratios, pips) live in
``internal.feature_engineering``; this module is the *weights* — the
action-type tier table and the per-row bonus coefficients — and the argmax.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from settlrl_engine.board.layout import N_TILES, N_VERTICES
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import KeyScalar
from settlrl_engine.env import (
    N_FLAT,
    ActionType,
    Observation,
)
from settlrl_search.policy import FlatAction, FlatMask
from settlrl_search.priors import TIER_SCORES
from settlrl_search.rows import ROW_PARAMS, ROW_TYPE

from settlrl_agents.internal.feature_engineering import (
    maritime_ratio,
    target_build,
    tile_pips,
    vertex_pips,
)

_ROW_IDX = ROW_PARAMS.idx
_ROW_TARGET = ROW_PARAMS.target

# Row groups whose bonus is target-dependent.
_VERTEX_BUILD = (
    (ROW_TYPE == ActionType.SETUP_SETTLEMENT)
    | (ROW_TYPE == ActionType.BUILD_SETTLEMENT)
    | (ROW_TYPE == ActionType.BUILD_CITY)
)
_ROBBER_MOVE = (ROW_TYPE == ActionType.MOVE_ROBBER) | (
    ROW_TYPE == ActionType.PLAY_KNIGHT
)
_DISCARD = ROW_TYPE == ActionType.DISCARD
_MARITIME = ROW_TYPE == ActionType.MARITIME_TRADE
_ACCEPT = ROW_TYPE == ActionType.ACCEPT_TRADE
_REJECT = ROW_TYPE == ActionType.REJECT_TRADE

_RES_IDX = jnp.clip(_ROW_IDX, 0, N_RESOURCES - 1)
_RES_TARGET = jnp.clip(_ROW_TARGET, 0, N_RESOURCES - 1)


def greedy_policy(key: KeyScalar, obs: Observation, mask: FlatMask) -> FlatAction:
    """Highest-priority legal action, ties broken uniformly at random.

    Priorities: city > settlement > dev card > road > play dev > everything
    forced (roll/setup road/discard/robber/trade response) > end turn. Trade
    sense comes from the target-build features (needs and surplus):

    - maritime trades run only when productive (the bought card is needed,
      the sold cards are pure surplus), preferring the scarcest need;
    - a domestic offer is accepted exactly when it is paid entirely from
      surplus and either advances a need or consolidates (more of the paid
      card held than of the received one); it never offers its own;
    - the discard gives up surplus before anything else, most-held first.

    Within a tier, settlement/city targets are scored by adjacent-tile pips
    and the robber goes to the highest-pip tile (preferring a steal).
    """
    v_pips = vertex_pips(obs["tile_number"])
    t_pips = tile_pips(obs["tile_number"])
    held = obs["self_resources"].astype(jnp.float32)

    _, need, surplus = target_build(obs)
    ratio = maritime_ratio(obs)

    # A maritime row sells _RES_IDX for _RES_TARGET: productive iff the buy is
    # needed and the sale never dips into the target's own ingredients. The
    # +150 gate lifts exactly those rows over END_TURN (100).
    productive = (need[_RES_TARGET] > 0) & (surplus[_RES_IDX] >= ratio[_RES_IDX])
    maritime = 150.0 * productive + need[_RES_TARGET]

    # The pending trade from the responder's side: it would *get* the
    # proposer's give bundle and *pay* the asked-for bundle. Accept iff the
    # payment is pure surplus and the trade either advances a need or
    # consolidates surplus toward scarcity; the 2x bonus against the fixed
    # reject bonus of 1 keeps the response deterministic under the <1 noise.
    get = obs["trade_give"].astype(jnp.float32)
    pay = obs["trade_receive"].astype(jnp.float32)
    pays_surplus = jnp.all(pay <= surplus)
    advances = (need * get).sum() >= 1.0
    consolidates = (held * pay).sum() - (held * get).sum() >= 1.0
    accept = pays_surplus & (advances | consolidates)

    bonus = jnp.where(
        _VERTEX_BUILD,
        v_pips[jnp.clip(_ROW_IDX, 0, N_VERTICES - 1)],
        jnp.where(
            _ROBBER_MOVE,
            t_pips[jnp.clip(_ROW_IDX, 0, N_TILES - 1)] + (_ROW_TARGET >= 0),
            jnp.where(
                _DISCARD,
                held[_RES_IDX] + 12.0 * (surplus[_RES_IDX] > 0),
                jnp.where(
                    _MARITIME,
                    maritime,
                    jnp.where(_ACCEPT, 2.0 * accept, jnp.where(_REJECT, 1.0, 0.0)),
                ),
            ),
        ),
    )
    noise = jax.random.uniform(key, (N_FLAT,))
    score = TIER_SCORES + bonus + noise
    return jnp.argmax(jnp.where(mask, score, -jnp.inf))
