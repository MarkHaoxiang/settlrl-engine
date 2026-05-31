"""Robber rules: who can be stolen from, and the random steal (single-game)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.board.layout import TILE_V
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import BoardState, IntScalar, PlayerMaskVec


def steal(state: BoardState, thief: IntScalar, victim: IntScalar) -> BoardState:
    """Move one random resource card from ``victim`` to ``thief`` (no-op if empty)."""
    res = state.player_resources.astype(jnp.int32)
    hand = res[victim]  # (R,)
    total = hand.sum()
    key, sub = jax.random.split(state.key)
    probs = jnp.where(
        total > 0,
        hand / jnp.maximum(total, 1),
        jnp.full((N_RESOURCES,), 1.0 / N_RESOURCES),
    )
    choice = jax.random.choice(sub, N_RESOURCES, p=probs)
    do = (total > 0).astype(jnp.int32)
    res = res.at[victim, choice].add(-do)
    res = res.at[thief, choice].add(do)
    return state._replace(player_resources=res.astype(jnp.uint8), key=key)


def robber_victim_mask(
    state: BoardState, tile: IntScalar, current: IntScalar
) -> PlayerMaskVec:
    """(N_PLAYERS,) bool: players != current with a building on ``tile`` and cards."""
    o = state.vertex_owner[TILE_V[tile]]  # (6,)
    pl = jnp.clip(o.astype(jnp.int32) - 1, 0, N_PLAYERS - 1)
    present = jnp.zeros((N_PLAYERS,), jnp.bool_).at[pl].max(o > 0)
    has_cards = state.player_resources.astype(jnp.int32).sum(axis=1) > 0
    return present & has_cards & (jnp.arange(N_PLAYERS) != current)
