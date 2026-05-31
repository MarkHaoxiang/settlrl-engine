"""Top-level Board type and convenience helpers for constructing / shaping a
batched game state. These are deliberately action-agnostic (they only touch
layout + state arrays) so that ``board.py`` stays free of import cycles with the
action layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.board.layout import BoardLayout, desert_tile, make_layout
from catan_engine.board.state import BoardState, GamePhase, make_board_state

Board = tuple[BoardLayout, BoardState]


def make_board(batch_size: int = 1, seed: int = 0) -> Board:
    """A fresh batched Board: random layout paired with a setup-phase state.

    The robber starts on the desert tile (rulebook); since tile positions are
    randomised it is read off the generated layout.
    """
    key = jax.random.key(seed)
    layout = make_layout(batch_size, key=key)
    state = make_board_state(batch_size, key=key)
    state = state._replace(robber=desert_tile(layout.tile_resource))
    return layout, state


def replicate(board: Board, batch_size: int) -> Board:
    """Tile a single-game (batch=1) board into shape ``(batch_size, ...)``."""

    def tile(x: jax.Array) -> jax.Array:
        return jnp.broadcast_to(x, (batch_size,) + x.shape[1:])

    return (
        jax.tree_util.tree_map(tile, board[0]),
        jax.tree_util.tree_map(tile, board[1]),
    )


def set_phase(board: Board, phase: GamePhase, lane: int = 0) -> Board:
    layout, state = board
    return (layout, state._replace(phase=state.phase.at[lane].set(int(phase))))


def to_main(board: Board, lane: int = 0, player: int = 0) -> Board:
    """Force a MAIN-phase, post-roll state for ``player`` in ``lane``."""
    layout, state = board
    state = state._replace(
        phase=state.phase.at[lane].set(int(GamePhase.MAIN)),
        current_player=state.current_player.at[lane].set(player),
        has_rolled=state.has_rolled.at[lane].set(1),
    )
    return (layout, state)


def give(board: Board, player: int, resources: list[int], lane: int = 0) -> Board:
    """Set ``player``'s resource hand in ``lane`` (order [sheep,wheat,wood,brick,ore])."""
    layout, state = board
    row = jnp.asarray(resources, dtype=state.player_resources.dtype)
    return (
        layout,
        state._replace(
            player_resources=state.player_resources.at[lane, player].set(row)
        ),
    )


def place_settlement(board: Board, player: int, vertex: int, lane: int = 0) -> Board:
    """Directly place ``player``'s settlement at ``vertex`` (grants +1 building VP)."""
    layout, state = board
    state = state._replace(
        vertex_owner=state.vertex_owner.at[lane, vertex].set(player + 1),
        vertex_type=state.vertex_type.at[lane, vertex].set(1),
        victory_points=state.victory_points.at[lane, player].add(1),
    )
    return (layout, state)


def place_road(board: Board, player: int, edge: int, lane: int = 0) -> Board:
    """Directly place ``player``'s road on ``edge``."""
    layout, state = board
    return (
        layout,
        state._replace(edge_road=state.edge_road.at[lane, edge].set(player + 1)),
    )


def place_city(board: Board, player: int, vertex: int, lane: int = 0) -> Board:
    """Directly place ``player``'s city at ``vertex`` (worth +2 building VP)."""
    layout, state = board
    had = state.vertex_type[lane, vertex] == 1  # already a settlement here?
    gain = jnp.where(had, 1, 2).astype(state.victory_points.dtype)
    state = state._replace(
        vertex_owner=state.vertex_owner.at[lane, vertex].set(player + 1),
        vertex_type=state.vertex_type.at[lane, vertex].set(2),
        victory_points=state.victory_points.at[lane, player].add(gain),
    )
    return (layout, state)


def give_dev_card(
    board: Board, player: int, card: int, count: int = 1, lane: int = 0
) -> Board:
    """Add ``count`` copies of dev ``card`` to ``player``'s hand in ``lane``."""
    layout, state = board
    return (
        layout,
        state._replace(dev_hand=state.dev_hand.at[lane, player, card].add(count)),
    )


def set_robber(board: Board, tile: int, lane: int = 0) -> Board:
    """Place the robber on ``tile`` in ``lane``."""
    layout, state = board
    return (layout, state._replace(robber=state.robber.at[lane].set(tile)))
