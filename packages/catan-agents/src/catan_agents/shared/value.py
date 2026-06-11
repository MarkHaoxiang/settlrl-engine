"""State value functions: how good is this board for a given player?

A :class:`ValueFunction` scores a concrete board. The search agents only ever
hand it *sampled* worlds (see ``shared.sample``), so the "hidden" fields it
reads there are belief-consistent samples; it still treats opponents' dev
cards as counts (their composition is a distribution over the known deck) and
the player's own as exact.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from catan_engine.board.dev_cards import DEV_CARD_COST, DEV_CARD_COUNTS, DevCard
from catan_engine.board.layout import (
    EDGE_V,
    N_TILES,
    N_VERTICES,
    PORT_V,
    TILE_V,
    BoardLayout,
)
from catan_engine.board.resources import CITY_COST, SETTLEMENT_COST
from catan_engine.board.state import (
    MAX_SETTLEMENTS,
    SETTLEMENT,
    BoardState,
    IntScalar,
)
from jaxtyping import Array, Float

Value = Float[Array, ""]
"""A scalar state score for one player: higher is better, arbitrary scale."""


@runtime_checkable
class ValueFunction(Protocol):
    """A single-game state evaluation, pure and ``jit`` / ``vmap`` compatible.

    ``layout`` / ``state`` are one game's board (no batch axis); returns the
    state's value from ``player``'s point of view.
    """

    def __call__(
        self, layout: BoardLayout, state: BoardState, player: IntScalar
    ) -> Value: ...


def tile_pips(tile_number: jax.Array) -> Float[Array, f"tiles={N_TILES}"]:
    """Expected-production weight per tile: 6 - |7 - number| (0 for the desert)."""
    n = tile_number.astype(jnp.int32)
    return jnp.where(n == 0, 0, 6 - jnp.abs(7 - n)).astype(jnp.float32)


def vertex_pips(tile_number: jax.Array) -> Float[Array, f"vertices={N_VERTICES}"]:
    """Summed pips of each vertex's adjacent tiles."""
    pips = tile_pips(tile_number)
    acc = jnp.zeros((N_VERTICES,), jnp.float32)
    return acc.at[TILE_V.reshape(-1)].add(jnp.repeat(pips, TILE_V.shape[1]))


_VP_CARD_SHARE = float(DEV_CARD_COUNTS[DevCard.VICTORY_POINT]) / float(
    sum(DEV_CARD_COUNTS)
)
_SETTLEMENT_COST_ARR = jnp.asarray(SETTLEMENT_COST, jnp.float32)
_CITY_COST_ARR = jnp.asarray(CITY_COST, jnp.float32)
_DEV_COST_ARR = jnp.asarray(DEV_CARD_COST, jnp.float32)


def make_heuristic(
    *,
    w_vp: float = 10.0,
    w_prod: float = 1.0,
    w_hand: float = 0.3,
    w_over: float = 0.4,
    w_dev: float = 1.5,
    w_spot: float = 0.5,
    w_road: float = 0.15,
    w_prog: float = 2.0,
    w_knight: float = 0.5,
    w_diverse: float = 0.6,
    w_port: float = 0.0,
    w_wheat_ore: float = 0.25,
    w_race: float = 0.8,
) -> ValueFunction:
    """Build a weighted heuristic value function (see :func:`heuristic_value`).

    Terms per player: total VP (awards and VP cards included; opponents' VP
    cards by their deck-share prior), pip-weighted production of own buildings
    (robber-aware, city double), distinct resource types produced, hand
    diversity (sqrt per type) with a discard-risk penalty per card over seven,
    held dev cards, expansion (pips of the best settlement spot buildable right
    now, and own roads), completeness of the closest affordable build, knights
    played toward Largest Army, a wheat/ore production premium, and a
    superlinear closing-urgency term above six VP. The value is the player's
    strength minus the best opponent's.
    """

    def strength(
        layout: BoardLayout, state: BoardState, p: jax.Array, exact_dev: jax.Array
    ) -> jax.Array:
        vp = state.victory_points[p].astype(jnp.float32)
        vp += 2.0 * (state.longest_road_owner == p)
        vp += 2.0 * (state.largest_army_owner == p)
        # Production: per-tile pips of own buildings (city counts twice),
        # robber tile blocked; aggregated per resource for the diversity term.
        pips = tile_pips(layout.tile_number)
        pips = pips * (jnp.arange(N_TILES) != state.robber)
        weight = (
            ((state.vertex_owner[TILE_V] == p + 1) * state.vertex_type[TILE_V])
            .sum(axis=1)
            .astype(jnp.float32)
        )
        per_tile = pips * weight  # (T,)
        per_res = (
            jnp.zeros((5,), jnp.float32)
            .at[layout.tile_resource.astype(jnp.int32) % 5]  # desert pips are 0
            .add(per_tile)
        )
        production = per_res.sum()
        diversity = (per_res > 0).sum().astype(jnp.float32)

        res = state.player_resources[p].astype(jnp.float32)
        hand = jnp.sqrt(res).sum()
        over = jnp.maximum(res.sum() - 7.0, 0.0)

        n_dev = state.dev_hand[p].astype(jnp.float32).sum()
        own_vp_cards = state.dev_hand[p, DevCard.VICTORY_POINT].astype(jnp.float32)
        dev_vp = jnp.where(exact_dev, own_vp_cards, n_dev * _VP_CARD_SHARE)

        # Expansion: settlement spots buildable right now (empty, distance
        # rule, touching an own road) — what makes a road worth its cost.
        own_road = state.edge_road == p + 1
        occ = state.vertex_owner > 0
        u, v = EDGE_V[:, 0], EDGE_V[:, 1]
        nb_occ = jnp.zeros((N_VERTICES,), bool).at[u].max(occ[v]).at[v].max(occ[u])
        touched = jnp.zeros((N_VERTICES,), bool).at[u].max(own_road).at[v].max(own_road)
        is_settlement = (state.vertex_owner == p + 1) & (
            state.vertex_type == SETTLEMENT
        )
        in_stock = is_settlement.sum() < MAX_SETTLEMENTS
        spot = ~occ & ~nb_occ & touched & in_stock
        best_spot = jnp.max(jnp.where(spot, vertex_pips(layout.tile_number), 0.0))
        n_roads = own_road.sum().astype(jnp.float32)

        # Progress toward the closest next build (gated on it being usable).
        def completeness(cost: jax.Array) -> jax.Array:
            return jnp.minimum(res, cost).sum() / cost.sum()

        deck_left = state.dev_deck.astype(jnp.int32).sum() > 0
        progress = jnp.max(
            jnp.stack(
                [
                    completeness(_SETTLEMENT_COST_ARR) * jnp.any(spot),
                    completeness(_CITY_COST_ARR) * jnp.any(is_settlement),
                    completeness(_DEV_COST_ARR) * deck_left,
                ]
            )
        )

        knights = jnp.minimum(state.knights_played[p].astype(jnp.float32), 3.0)

        # Ports: a 2:1 port is worth the production it can convert; a 3:1
        # port a fraction of all production.
        port_alloc = layout.port_allocation.astype(jnp.int32)
        on_port = (state.vertex_owner[PORT_V] == p + 1).any(axis=1)  # (P_ports,)
        has_2to1 = (
            jnp.zeros((5,), bool).at[port_alloc % 5].max(on_port & (port_alloc < 5))
        )
        has_3to1 = jnp.any(on_port & (port_alloc == 5))
        ports = (has_2to1 * per_res).sum() + 0.3 * has_3to1 * production

        # Closing urgency: VPs matter superlinearly near the win.
        race = jnp.maximum(vp + dev_vp - 6.0, 0.0) ** 2

        return (
            w_vp * (vp + dev_vp)
            + w_prod * production
            + w_diverse * diversity
            + w_hand * hand
            - w_over * over
            + w_dev * n_dev
            + w_spot * best_spot
            + w_road * n_roads
            + w_prog * progress
            + w_knight * knights
            + w_port * ports
            + w_wheat_ore * (per_res[1] + per_res[4])
            + w_race * race
        )

    def value(layout: BoardLayout, state: BoardState, player: IntScalar) -> Value:
        players = jnp.arange(state.n_players)
        strengths = jax.vmap(lambda q: strength(layout, state, q, q == player))(players)
        mine = strengths[player]
        best_other = jnp.max(jnp.where(players == player, -jnp.inf, strengths))
        return mine - best_other

    return value


heuristic_value = make_heuristic()
"""The shipped heuristic, at the weights that won the 2-player CLI tournament."""
