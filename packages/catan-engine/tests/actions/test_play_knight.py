"""Tests for the vectorized PlayKnight action."""

import jax.numpy as jnp
import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    place_settlement,
    set_robber,
    to_main,
)
from catan_engine.action_vec import PlayKnight
from catan_engine.dev_cards import DevCard
from catan_engine.rules_vec import TILE_V
from tests.actions.fixtures import fmt

_TILE_V = np.asarray(TILE_V)


def _knight_fixture(tile: int = 0) -> Board:
    """MAIN board: player 0 holds a Knight; player 1 sits on ``tile`` with 1 sheep."""
    board = to_main(make_board(seed=0))
    board = give_dev_card(board, 0, DevCard.KNIGHT)
    v = int(_TILE_V[tile, 0])
    board = place_settlement(board, 1, v)
    board = give(board, 1, [1, 0, 0, 0, 0])  # 1 sheep to steal
    board = set_robber(board, (tile + 1) % _TILE_V.shape[0])
    return board


class TestPlayKnight(TestCase):
    def test_success(self) -> None:
        T = 0
        board = _knight_fixture(T)
        state, result = PlayKnight()(board, (jnp.array([T]), jnp.array([1])))
        self.assertExpectedInline(
            fmt(
                result,
                robber=int(state.robber[0]),
                knights=int(state.knights_played[0, 0]),
                dev_played=int(state.dev_played[0]),
                p0_sheep=int(state.player_resources[0, 0, 0]),
                p1_sheep=int(state.player_resources[0, 1, 0]),
            ),
            """\
result=OK
robber=0
knights=1
dev_played=1
p0_sheep=1
p1_sheep=0""",
        )

    def test_no_victim(self) -> None:
        # A tile with no opponent buildings: move the robber, steal from no one.
        T = 0
        board = to_main(make_board(seed=0))
        board = give_dev_card(board, 0, DevCard.KNIGHT)
        board = set_robber(board, (T + 1) % _TILE_V.shape[0])
        before = np.asarray(board[1].player_resources)
        state, result = PlayKnight()(board, (jnp.array([T]), jnp.array([-1])))
        assert int(result[0]) == ActionResult.SUCCESS.value
        assert int(state.robber[0]) == T
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_no_knight(self) -> None:
        T = 0
        board = to_main(make_board(seed=0))
        board = place_settlement(board, 1, int(_TILE_V[T, 0]))
        board = give(board, 1, [1, 0, 0, 0, 0])
        board = set_robber(board, (T + 1) % _TILE_V.shape[0])
        before = np.asarray(board[1].player_resources)
        state, result = PlayKnight()(board, (jnp.array([T]), jnp.array([1])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_tile_is_robber(self) -> None:
        T = 0
        board = _knight_fixture(T)
        board = set_robber(board, T)  # robber already on T
        before = np.asarray(board[1].player_resources)
        state, result = PlayKnight()(board, (jnp.array([T]), jnp.array([1])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)

    def test_invalid_out_of_range_tile(self) -> None:
        board = _knight_fixture(0)
        state, result = PlayKnight()(board, (jnp.array([999]), jnp.array([1])))
        assert int(result[0]) == ActionResult.INVALID.value

    def test_invalid_dev_already_played(self) -> None:
        T = 0
        board = _knight_fixture(T)
        layout, st = board
        board = (layout, st._replace(dev_played=st.dev_played.at[0].set(1)))
        before = np.asarray(board[1].player_resources)
        state, result = PlayKnight()(board, (jnp.array([T]), jnp.array([1])))
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.player_resources), before)
