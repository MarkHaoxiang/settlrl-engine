"""The 7-resolution mechanics: discarding, moving the robber, and stealing.

Holds the steal / victim-mask primitives, the shared robber-victim validation
reused by ``MoveRobber`` and ``PlayKnight`` (the latter lives in
``development``), and the ``MoveRobber`` / ``Discard`` action cores (the
latter discards one card per action -- see the Discard section). All
single-game and traceable.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board import Board
from catan_engine.board.layout import N_TILES, TILE_V, BoardLayout
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import (
    BoardState,
    BoolScalar,
    GamePhase,
    IntScalar,
    PlayerMaskVec,
    to_u8,
    tree_select,
)
from catan_engine.mechanics.common import (
    INVALID,
    SUCCESS,
    IndexParam,
    Mask,
    ResultCode,
    TwoIndexParams,
    agent_selection_single,
)


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


# ===========================================================================
# Shared robber-targeting helpers (reused by MoveRobber and PlayKnight)
# ===========================================================================


def valid_robber_victim(
    state: BoardState, tile: jax.Array, player: jax.Array, victim: IntScalar
) -> BoolScalar:
    """Victim choice is legal for a robber move onto ``tile`` by ``player``.

    If any opponent can be robbed on ``tile``, ``victim`` must name one of them;
    otherwise the only legal choice is ``-1`` ("steal from no one").
    """
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    mask = robber_victim_mask(state, tile, player)
    victims_exist = jnp.any(mask)
    return jnp.where(
        victims_exist,
        (victim >= 0) & (victim < N_PLAYERS) & mask[vc],
        victim == -1,
    )


def apply_steal(state: BoardState, player: jax.Array, victim: IntScalar) -> BoardState:
    """Steal a random card from ``victim`` when ``victim >= 0``; else leave state."""
    vc = jnp.clip(victim, 0, N_PLAYERS - 1)
    stolen = steal(state, player, vc)
    return tree_select(victim >= 0, stolen, state)


# ===========================================================================
# MoveRobber
# ===========================================================================


def _move_robber_avail(
    layout: BoardLayout, state: BoardState, params: tuple[IntScalar, IntScalar]
) -> BoolScalar:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    phase_ok = state.phase == GamePhase.MOVE_ROBBER
    tile_in_range = (tile >= 0) & (tile < N_TILES)
    tile_moves = tile != state.robber
    valid_victim = valid_robber_victim(state, t, player, victim)
    return phase_ok & tile_in_range & tile_moves & valid_victim


def _move_robber_apply(
    layout: BoardLayout,
    state: BoardState,
    params: tuple[IntScalar, IntScalar],
    available: BoolScalar,
) -> tuple[BoardState, IntScalar]:
    tile, victim = params
    player = state.current_player.astype(jnp.int32)
    t = jnp.clip(tile, 0, N_TILES - 1)
    # Knight-before-roll resumes ROLL; the post-7 robber move resumes MAIN.
    new_phase = jnp.where(
        state.has_rolled != 0, GamePhase.MAIN, GamePhase.ROLL
    ).astype(jnp.uint8)
    cand = state._replace(
        robber=t.astype(state.robber.dtype),
        phase=new_phase,
    )
    cand = apply_steal(cand, player, victim)
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_move_robber_avail_b = jax.jit(jax.vmap(_move_robber_avail))
_move_robber_apply_b = jax.jit(jax.vmap(_move_robber_apply))


def move_robber_available(board: Board, params: TwoIndexParams) -> Mask:
    """``(batch,)`` legality of a (tile, victim) robber move (no state change)."""
    return cast(Mask, _move_robber_avail_b(board[0], board[1], params))


def move_robber_step(
    board: Board, params: TwoIndexParams
) -> tuple[BoardState, ResultCode]:
    """Move the robber to ``tile`` and steal from ``victim`` (``-1`` = no one).

    Resolves the post-7 (or knight-before-roll) robber move; never wins.
    """
    available = _move_robber_avail_b(board[0], board[1], params)
    return cast(
        "tuple[BoardState, ResultCode]",
        _move_robber_apply_b(board[0], board[1], params, available),
    )


# ===========================================================================
# Discard
# ===========================================================================
#
# One card per action: the acting discarder (the lowest-indexed player still
# owing cards -- ``agent_selection_single``) gives up a single card of the
# chosen resource, decrementing its owed count. The phase advances to
# MOVE_ROBBER once every owed count is zero. Splitting the discard into
# per-card moves keeps the choice space flat (one action per resource) instead
# of the combinatorial space of whole-hand splits.


def _discard_avail(
    layout: BoardLayout, state: BoardState, params: IntScalar
) -> BoolScalar:
    resource = params
    r = jnp.clip(resource, 0, N_RESOURCES - 1)
    p = agent_selection_single(state)
    phase_ok = state.phase == GamePhase.DISCARD
    in_range = (resource >= 0) & (resource < N_RESOURCES)
    owes = state.pending_discard[p] > 0
    holds = state.player_resources[p, r] > 0
    return phase_ok & in_range & owes & holds


def _discard_apply(
    layout: BoardLayout,
    state: BoardState,
    params: IntScalar,
    available: BoolScalar,
) -> tuple[BoardState, IntScalar]:
    resource = params
    r = jnp.clip(resource, 0, N_RESOURCES - 1)
    p = agent_selection_single(state)
    new_resources = to_u8(state.player_resources.astype(jnp.int32).at[p, r].add(-1))
    new_pending = state.pending_discard.astype(jnp.int32).at[p].add(-1)
    new_phase = jnp.where(
        new_pending.sum() == 0, GamePhase.MOVE_ROBBER, GamePhase.DISCARD
    ).astype(state.phase.dtype)
    cand = state._replace(
        player_resources=new_resources,
        pending_discard=to_u8(new_pending),
        phase=new_phase,
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_discard_avail_b = jax.jit(jax.vmap(_discard_avail))
_discard_apply_b = jax.jit(jax.vmap(_discard_apply))


def discard_available(board: Board, params: IndexParam) -> Mask:
    """``(batch,)`` legality of discarding one card of ``resource`` (no state change)."""
    return cast(Mask, _discard_avail_b(board[0], board[1], params))


def discard_step(board: Board, params: IndexParam) -> tuple[BoardState, ResultCode]:
    """Discard one card of ``resource`` from the acting discarder (post-7).

    Discarding is sequential, one card per step: the acting discarder is the
    lowest-indexed player still owing cards, and each action decrements its
    owed count by one. When every owed count reaches zero the phase advances
    to MOVE_ROBBER. Never wins.
    """
    available = _discard_avail_b(board[0], board[1], params)
    return cast(
        "tuple[BoardState, ResultCode]",
        _discard_apply_b(board[0], board[1], params, available),
    )
