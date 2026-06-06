"""Tests for the vectorized SetupRoad action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.setup import setup_road_step
from catan_engine.board import Board, place_road, set_phase
from catan_engine.board.layout import EDGE_V
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt

# First edge incident to vertex 0.
_E0 = int(np.where((np.asarray(EDGE_V) == 0).any(axis=1))[0][0])


def test_success(setup_road_board: Board, render: Callable[..., str]) -> None:
    state, result = setup_road_step(setup_road_board, jnp.array([_E0]))
    assert_expected_inline(
        fmt(
            result,
            edge_owner=int(state.edge_road[0, _E0]),
            setup_index=int(state.setup_index[0]),
            phase=str(GamePhase(int(state.phase[0]))),
            current_player=int(state.current_player[0]),
        ),
        """\
result=OK
edge_owner=1
setup_index=1
phase=SETUP_SETTLEMENT
current_player=1""",
    )
    assert_expected_inline(
        render(setup_road_board[0], state),
        r"""


          ORE             3:1
               /o\     /o\     /o\
              /   \   /   \   /   \
            o/     \o/     \o/     \o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |       |
           /o\     /o\     /o\     1o\   3:1
          /   \   /   \   /   \   1   \
        o/     \o/     \o/     \11     \o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\     /o\     /o\     /o\     /o\
      /   \   /   \   /   \   /   \   /   \
    o/     \o/     \o/     \o/     \o/     \o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  |
    |   8   |   4   |   3   |       |  12   |   3
    |       |       |       |  <R>  |       |
    o\     /o\     /o\     /o\     /o\     /o
      \   /   \   /   \   /   \   /   \   /
       \o/     \o/     \o/     \o/     \o/
        |  SHP  |  ORE  |  BRK  |  BRK  |
        |   8   |   3   |  11   |   6   |
  3:1   |       |       |       |       |
        o\     /o\     /o\     /o\     /o
          \   /   \   /   \   /   \   /
           \o/     \o/     \o/     \o/   BRK
            |  WHT  |  WHT  |  WOD  |
            |   4   |   9   |   5   |
            |       |       |       |
            o\     /o\     /o\     /o
              \   /   \   /   \   /
               \o/     \o/     \o/
          SHP             WHT


""",
    )


def test_setup_complete(setup_road_board: Board) -> None:
    layout, st = setup_road_board
    n_setup = 2 * int(st.n_players[0])  # the snake places 2 settlements each
    st = st._replace(setup_index=st.setup_index.at[0].set(n_setup - 1))
    state, result = setup_road_step((layout, st), jnp.array([_E0]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.setup_index[0]) == n_setup
    assert int(state.phase[0]) == GamePhase.ROLL
    assert int(state.current_player[0]) == 0


def test_invalid_wrong_phase(setup_road_board: Board) -> None:
    board = set_phase(setup_road_board, GamePhase.ROLL)
    before = np.asarray(board[1].edge_road)
    state, result = setup_road_step(board, jnp.array([_E0]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.edge_road), before)


def test_invalid_edge_occupied(setup_road_board: Board) -> None:
    board = place_road(setup_road_board, 0, _E0)
    before = np.asarray(board[1].edge_road)
    state, result = setup_road_step(board, jnp.array([_E0]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.edge_road), before)


def test_invalid_does_not_touch_settlement(setup_road_board: Board) -> None:
    # An edge with no endpoint owned by the player.
    edges = np.asarray(EDGE_V)
    far = int(np.where(~(edges == 0).any(axis=1))[0][0])
    before = np.asarray(setup_road_board[1].edge_road)
    state, result = setup_road_step(setup_road_board, jnp.array([far]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.edge_road), before)
