"""Initial-placement (setup phase) rules: turn order and the 2nd-settlement grant."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.geometry import NO_IDX, V_TILES
from catan_engine.layout import MAX_VERTEX_DEGREE, N_TILES, BoardLayout
from catan_engine.resources import BANK_INITIAL, N_PLAYERS
from catan_engine.state import BoardState
from catan_engine.tile import Tile

# Setup placement order over 2 * N_PLAYERS settlements (snake / boustrophedon),
# as a traceable int32 array so SetupRoad can advance the turn branchlessly.
SETUP_ORDER = list(range(N_PLAYERS)) + list(range(N_PLAYERS - 1, -1, -1))
SETUP_ORDER_ARR = jnp.array(SETUP_ORDER, dtype=jnp.int32)  # (2 * N_PLAYERS,)
N_SETUP = len(SETUP_ORDER)


def grant_setup_resources(
    layout: BoardLayout, state: BoardState, vertex: jax.Array, player: jax.Array
) -> BoardState:
    """Grant one resource per (non-desert) tile adjacent to a 2nd settlement."""
    res = state.player_resources.astype(jnp.int32)
    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    tiles = V_TILES[vertex]  # (MAX_VERTEX_DEGREE,)
    for i in range(MAX_VERTEX_DEGREE):
        t = tiles[i]
        t_c = jnp.clip(t, 0, N_TILES - 1)
        resource = layout.tile_resource[t_c].astype(jnp.int32)
        ok = (t != NO_IDX) & (resource != Tile.DESERT) & (bank[resource] > 0)
        add = ok.astype(jnp.int32)
        res = res.at[player, resource].add(add)
        bank = bank.at[resource].add(-add)
    return state._replace(player_resources=res.astype(jnp.uint8))
