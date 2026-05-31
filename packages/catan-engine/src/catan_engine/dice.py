"""Dice rolling and the resource production it triggers (single-game, traceable)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.layout import N_TILES, TILE_V, BoardLayout
from catan_engine.resources import BANK_INITIAL, N_PLAYERS, N_RESOURCES
from catan_engine.state import BoardState, IntScalar, KeyScalar
from catan_engine.tile import Tile


def roll_dice(key: KeyScalar) -> tuple[KeyScalar, IntScalar]:
    """Return (advanced key, two-dice sum 2..12)."""
    key, k1, k2 = jax.random.split(key, 3)
    d1 = jax.random.randint(k1, (), 1, 7)
    d2 = jax.random.randint(k2, (), 1, 7)
    return key, (d1 + d2).astype(jnp.int32)


def distribute_resources(
    layout: BoardLayout, state: BoardState, roll: IntScalar
) -> BoardState:
    """Pay out resources for ``roll`` to building owners, honouring the bank.

    Bank rule: if demand for a resource exceeds the bank and more than one
    player is owed it, no one receives it; if exactly one player is owed it,
    they receive whatever the bank has left.
    """
    owner = state.vertex_owner
    kind = state.vertex_type
    res = state.player_resources.astype(jnp.int32)  # (P, R)

    produces = (
        (layout.tile_number == roll)
        & (jnp.arange(N_TILES) != state.robber)
        & (layout.tile_resource != Tile.DESERT)
    )  # (N_TILES,)

    c_owner = owner[TILE_V]  # (N_TILES, 6)
    c_kind = kind[TILE_V]
    amt = jnp.where(c_kind == 1, 1, jnp.where(c_kind == 2, 2, 0)) * produces[:, None]
    amt = jnp.where(c_owner > 0, amt, 0).astype(jnp.int32)
    pl = jnp.clip(c_owner.astype(jnp.int32) - 1, 0, N_PLAYERS - 1)
    res_idx = jnp.broadcast_to(
        layout.tile_resource[:, None].astype(jnp.int32), (N_TILES, 6)
    )
    gains = jnp.zeros((N_PLAYERS, N_RESOURCES), jnp.int32).at[
        pl.reshape(-1), res_idx.reshape(-1)
    ].add(amt.reshape(-1))

    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    total = gains.sum(axis=0)  # (R,)
    n_claim = (gains > 0).sum(axis=0)  # (R,)
    enough = total <= bank
    single = n_claim == 1
    granted = jnp.where(enough, gains, jnp.where(single, jnp.minimum(gains, bank), 0))
    new_res = (res + granted).astype(jnp.uint8)
    return state._replace(player_resources=new_res)
