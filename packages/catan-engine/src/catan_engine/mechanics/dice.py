"""Dice rolling and the resource production it triggers (single-game, traceable).

Holds the ``RollDice`` action core (avail/apply) alongside the ``roll_dice`` /
``distribute_resources`` helpers it composes.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board import Board
from catan_engine.board.layout import N_TILES, TILE_V, BoardLayout
from catan_engine.board.resources import BANK_INITIAL, N_RESOURCES
from catan_engine.board.state import (
    CITY,
    SETTLEMENT,
    BoardState,
    BoolScalar,
    GamePhase,
    IntScalar,
    KeyScalar,
    tree_select,
)
from catan_engine.board.tile import Tile
from catan_engine.mechanics.common import INVALID, SUCCESS, Mask, ResultCode


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
    amt = (
        jnp.where(c_kind == SETTLEMENT, 1, jnp.where(c_kind == CITY, 2, 0))
        * produces[:, None]
    )
    amt = jnp.where(c_owner > 0, amt, 0).astype(jnp.int32)
    pl = jnp.clip(c_owner.astype(jnp.int32) - 1, 0, state.n_players - 1)
    res_idx = jnp.broadcast_to(
        layout.tile_resource[:, None].astype(jnp.int32), (N_TILES, 6)
    )
    gains = jnp.zeros((state.n_players, N_RESOURCES), jnp.int32).at[
        pl.reshape(-1), res_idx.reshape(-1)
    ].add(amt.reshape(-1))

    bank = BANK_INITIAL - res.sum(axis=0)  # (R,)
    total = gains.sum(axis=0)  # (R,)
    n_claim = (gains > 0).sum(axis=0)  # (R,)
    enough = total <= bank
    single = n_claim == 1
    # When `single`, capping every player row at `bank` is safe: only the one
    # claimant has a nonzero `gains[:, r]`, so the others clip from 0 to 0.
    granted = jnp.where(enough, gains, jnp.where(single, jnp.minimum(gains, bank), 0))
    new_res = (res + granted).astype(jnp.uint8)
    return state._replace(player_resources=new_res)


# ===========================================================================
# RollDice action
# ===========================================================================


def _roll_avail(layout: BoardLayout, state: BoardState, params: None) -> BoolScalar:
    return (state.phase == GamePhase.ROLL) & (state.has_rolled == 0)


def _roll_apply(
    layout: BoardLayout, state: BoardState, params: None, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    key, roll = roll_dice(state.key)
    is_seven = roll == 7

    hand = state.player_resources.astype(jnp.int32).sum(axis=1)  # (P,)
    owes = jnp.where(hand > 7, hand // 2, 0).astype(jnp.uint8)
    pending = jnp.where(is_seven, owes, jnp.zeros_like(owes))
    any_discard = jnp.sum(pending) > 0
    phase_seven = jnp.where(any_discard, GamePhase.DISCARD, GamePhase.MOVE_ROBBER)
    new_phase = jnp.where(is_seven, phase_seven, GamePhase.MAIN).astype(jnp.uint8)

    distributed = distribute_resources(layout, state, roll)
    new_res = jnp.where(is_seven, state.player_resources, distributed.player_resources)

    cand = state._replace(
        key=key,
        dice_roll=roll.astype(jnp.uint8),
        has_rolled=jnp.uint8(1),
        phase=new_phase,
        pending_discard=pending,
        player_resources=new_res,
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_roll_avail_b = jax.jit(jax.vmap(_roll_avail, in_axes=(0, 0, None)))
_roll_apply_b = jax.jit(jax.vmap(_roll_apply, in_axes=(0, 0, None, 0)))


def roll_available(board: Board, params: None = None) -> Mask:
    """``(batch,)`` legality of RollDice per game (no state change)."""
    return cast(Mask, _roll_avail_b(board[0], board[1], None))


def roll_step(board: Board, params: None = None) -> tuple[BoardState, ResultCode]:
    """Apply RollDice per game; return (new state, ActionResult codes)."""
    available = _roll_avail_b(board[0], board[1], None)
    return cast(
        "tuple[BoardState, ResultCode]",
        _roll_apply_b(board[0], board[1], None, available),
    )
