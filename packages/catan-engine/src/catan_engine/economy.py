"""Costs, building counts, affordability, payment, bank stock and victory points.

Single-game, traceable helpers operating on the ``BoardState`` arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.dev_cards import DEV_CARD_COST, DevCard
from catan_engine.resources import BANK_INITIAL, CITY_COST, ROAD_COST, SETTLEMENT_COST
from catan_engine.state import MAX_ROADS, BoardState

# Build-cost vectors in resource order [sheep, wheat, wood, brick, ore].
ROAD_COST_ARR = jnp.array(ROAD_COST, dtype=jnp.int32)
SETTLEMENT_COST_ARR = jnp.array(SETTLEMENT_COST, dtype=jnp.int32)
CITY_COST_ARR = jnp.array(CITY_COST, dtype=jnp.int32)
DEV_CARD_COST_ARR = jnp.array(DEV_CARD_COST, dtype=jnp.int32)


def count_roads(edge_road: jax.Array, player: jax.Array) -> jax.Array:
    return jnp.sum(edge_road == player + 1).astype(jnp.int32)


def count_settlements(
    vertex_owner: jax.Array, vertex_type: jax.Array, player: jax.Array
) -> jax.Array:
    return jnp.sum((vertex_owner == player + 1) & (vertex_type == 1)).astype(jnp.int32)


def count_cities(
    vertex_owner: jax.Array, vertex_type: jax.Array, player: jax.Array
) -> jax.Array:
    return jnp.sum((vertex_owner == player + 1) & (vertex_type == 2)).astype(jnp.int32)


def roads_left(edge_road: jax.Array, player: jax.Array) -> jax.Array:
    return MAX_ROADS - count_roads(edge_road, player)


def to_u8(x: jax.Array) -> jax.Array:
    """Saturating cast to uint8 (clip to ``[0, 255]``)."""
    return jnp.clip(x, 0, 255).astype(jnp.uint8)


def can_afford(resources_row: jax.Array, cost_arr: jax.Array) -> jax.Array:
    """True if a single player's resource row covers ``cost_arr``."""
    return jnp.all(resources_row.astype(jnp.int32) >= cost_arr)


def pay(
    player_resources: jax.Array, player: jax.Array, cost_arr: jax.Array
) -> jax.Array:
    """Subtract ``cost_arr`` from ``player``'s row (clipped at 0), returning uint8."""
    updated = player_resources.astype(jnp.int32).at[player].add(-cost_arr)
    return to_u8(updated)


def bank_stock(player_resources: jax.Array, resource: jax.Array) -> jax.Array:
    held = player_resources[:, resource].astype(jnp.int32).sum()
    return BANK_INITIAL - held


def player_total_vp(state: BoardState, player: jax.Array) -> jax.Array:
    """Building VP + awards + hidden Victory Point cards for ``player``."""
    total = state.victory_points[player].astype(jnp.int32)
    total += jnp.where(state.longest_road_owner == player, 2, 0)
    total += jnp.where(state.largest_army_owner == player, 2, 0)
    total += state.dev_hand[player, DevCard.VICTORY_POINT].astype(jnp.int32)
    return total
