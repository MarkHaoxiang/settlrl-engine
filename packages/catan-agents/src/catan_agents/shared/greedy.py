"""A scripted greedy baseline: fixed action-type priorities, pip-weighted targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from catan_engine.board.dev_cards import DEV_CARD_COST
from catan_engine.board.layout import EDGE_V, N_TILES, N_VERTICES, PORT_V
from catan_engine.board.resources import CITY_COST, N_RESOURCES, SETTLEMENT_COST
from catan_engine.board.state import SETTLEMENT, KeyScalar
from catan_engine.env import (
    N_ACTION_TYPES,
    N_FLAT,
    ActionType,
    Observation,
    flat_to_action,
)

from catan_agents.shared.policy import FlatAction, FlatMask
from catan_agents.shared.value import tile_pips, vertex_pips

_SETTLEMENT_COST = jnp.asarray(SETTLEMENT_COST, jnp.float32)
_CITY_COST = jnp.asarray(CITY_COST, jnp.float32)
_DEV_COST = jnp.asarray(DEV_CARD_COST, jnp.float32)

# Static decode of every flat row: its action type and (idx, target) params.
_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))
_ROW_IDX = _ROW_PARAMS.idx
_ROW_TARGET = _ROW_PARAMS.target

# Priority tier per action type. Tiers are spaced so no per-target bonus
# (|bonus| < 50) can cross between them; types sharing a tier are never both
# the argmax (disjoint phases, or strict domination). The one exception is
# deliberate: a *productive* MARITIME_TRADE row carries a +150 gate that lifts
# it over END_TURN (see ``greedy_policy``); unproductive conversions stay
# below, and the greedy player never offers a domestic trade. ACCEPT_TRADE /
# REJECT_TRADE share a tier on purpose: their trade bonus decides the
# response.
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
_MARITIME = _ROW_TYPE == ActionType.MARITIME_TRADE
_ACCEPT = _ROW_TYPE == ActionType.ACCEPT_TRADE
_REJECT = _ROW_TYPE == ActionType.REJECT_TRADE

_RES_IDX = jnp.clip(_ROW_IDX, 0, N_RESOURCES - 1)
_RES_TARGET = jnp.clip(_ROW_TARGET, 0, N_RESOURCES - 1)


def greedy_policy(key: KeyScalar, obs: Observation, mask: FlatMask) -> FlatAction:
    """Highest-priority legal action, ties broken uniformly at random.

    Priorities: city > settlement > dev card > road > play dev > everything
    forced (roll/setup road/discard/robber/trade response) > end turn. Trade
    sense comes from a target build — city with a settlement to upgrade, else
    a settlement with a spot buildable right now, else a dev card — whose
    missing cards are the *needs* and whose excess holdings the *surplus*:

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
    me = obs["self"].astype(jnp.uint8) + 1

    # The target build: the next thing worth saving for, by the build
    # priorities above (a settlement spot must be buildable right now —
    # empty, distance rule, touching an own road).
    owner = obs["vertex_owner"]
    has_settlement = jnp.any((owner == me) & (obs["vertex_type"] == SETTLEMENT))
    own_road = obs["edge_road"] == me
    occ = owner > 0
    u, v = EDGE_V[:, 0], EDGE_V[:, 1]
    nb_occ = jnp.zeros((N_VERTICES,), bool).at[u].max(occ[v]).at[v].max(occ[u])
    touched = jnp.zeros((N_VERTICES,), bool).at[u].max(own_road).at[v].max(own_road)
    has_spot = jnp.any(~occ & ~nb_occ & touched)
    cost = jnp.where(
        has_settlement,
        _CITY_COST,
        jnp.where(has_spot, _SETTLEMENT_COST, _DEV_COST),
    )
    need = jnp.maximum(cost - held, 0.0)
    surplus = jnp.maximum(held - cost, 0.0)

    # Own maritime ratio per resource (2 at the matching port, 3 with a
    # generic port, else 4) — what a productive sale must come out of surplus.
    port_alloc = obs["port_allocation"].astype(jnp.int32)
    on_port = (owner[PORT_V] == me).any(axis=1)
    has_2to1 = (
        jnp.zeros((N_RESOURCES,), bool)
        .at[port_alloc % N_RESOURCES]
        .max(on_port & (port_alloc < N_RESOURCES))
    )
    has_3to1 = jnp.any(on_port & (port_alloc == N_RESOURCES))
    ratio = jnp.where(has_2to1, 2.0, jnp.where(has_3to1, 3.0, 4.0))

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
    score = _BASE + bonus + noise
    return jnp.argmax(jnp.where(mask, score, -jnp.inf))
