"""Initial-placement (setup phase) rules: turn order and the 2nd-settlement grant."""

from __future__ import annotations

import jax.numpy as jnp

from catan_engine.layout import TILE_V, BoardLayout
from catan_engine.resources import BANK_INITIAL, N_PLAYERS, N_RESOURCES
from catan_engine.state import BoardState, IntScalar
from catan_engine.tile import Tile

# Setup placement order over 2 * N_PLAYERS settlements (snake / boustrophedon),
# as a traceable int32 array so SetupRoad can advance the turn branchlessly.
SETUP_ORDER = list(range(N_PLAYERS)) + list(range(N_PLAYERS - 1, -1, -1))
SETUP_ORDER_ARR = jnp.array(SETUP_ORDER, dtype=jnp.int32)  # (2 * N_PLAYERS,)
N_SETUP = len(SETUP_ORDER)


def grant_setup_resources(
    layout: BoardLayout, state: BoardState, vertex: IntScalar, player: IntScalar
) -> BoardState:
    """Grant one resource per (non-desert) tile adjacent to a 2nd settlement."""
    res = state.player_resources.astype(jnp.int32)
    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    # Tiles touching ``vertex`` (a vertex appears in at most 3 tiles' corners).
    incident = (TILE_V == vertex).any(axis=1)  # (N_TILES,)
    resource = layout.tile_resource.astype(jnp.int32)  # (N_TILES,)
    produce = (incident & (layout.tile_resource != Tile.DESERT)).astype(jnp.int32)
    # Scatter demand per resource, then grant what the bank can cover (granting
    # min(demand, bank) matches the old per-tile sequential payout).
    demand = jnp.zeros((N_RESOURCES,), jnp.int32).at[resource].add(produce)
    res = res.at[player].add(jnp.minimum(demand, bank))
    return state._replace(player_resources=res.astype(jnp.uint8))
