"""Tests for the vectorized MoveRobber action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.common import ResultCode, TwoIndexParams
from catan_engine.mechanics.development import play_knight_step
from catan_engine.mechanics.robber import move_robber_step
from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    place_settlement,
    set_phase,
    set_robber,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.layout import TILE_V
from catan_engine.board.state import BoardState, GamePhase
from tests.mechanics.actions.fixtures import fmt

_TILE_V = np.asarray(TILE_V)

# A robber-targeting action's batched step: (board, (tile, victim)) -> (state, code).
RobberStep = Callable[[Board, TwoIndexParams], tuple[BoardState, ResultCode]]


def _robber_board() -> Board:
    """MOVE_ROBBER board: player 1 on tile 0 with a card; robber elsewhere."""
    board = to_main(make_board(seed=0))
    board = set_phase(board, GamePhase.MOVE_ROBBER)
    board = place_settlement(board, 1, int(_TILE_V[0, 0]))
    board = give(board, 1, [1, 0, 0, 0, 0])
    return set_robber(board, 1 % _TILE_V.shape[0])


def _knight_board() -> Board:
    """MAIN board: player 0 holds a Knight; player 1 on tile 0 with a card."""
    board = to_main(make_board(seed=0))
    board = give_dev_card(board, 0, DevCard.KNIGHT)
    board = place_settlement(board, 1, int(_TILE_V[0, 0]))
    board = give(board, 1, [1, 0, 0, 0, 0])
    return set_robber(board, 1 % _TILE_V.shape[0])


# Both MoveRobber and PlayKnight share robber-targeting params (tile, victim) and
# the same invalid-target rules; sweep both through the shared cases below.
_ROBBER_ACTIONS = [
    pytest.param(move_robber_step, _robber_board, id="move_robber"),
    pytest.param(play_knight_step, _knight_board, id="play_knight"),
]


def test_success(robber_board: Board, render: Callable[..., str]) -> None:
    # Robber starts on tile 1; move it to tile 0 and steal from player 1.
    state, result = move_robber_step(robber_board, (jnp.array([0]), jnp.array([1])))
    assert_expected_inline(
        fmt(
            result,
            robber=int(state.robber[0]),
            phase=str(GamePhase(int(state.phase[0]))),
            p0_sheep=int(state.player_resources[0, 0, 0]),
            p1_sheep=int(state.player_resources[0, 1, 0]),
        ),
        """\
result=OK
robber=0
phase=MAIN
p0_sheep=1
p1_sheep=0""",
    )
    assert_expected_inline(
        render(robber_board[0], state),
        r"""


          ORE             3:1
               /o\     /o\     /o\
              /   \   /   \   /   \
            o/     \o/     \o/     \o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |  <R>  |
           /o\     /o\     /o\     /o\   3:1
          /   \   /   \   /   \   /   \
        o/     \o/     \o/     \2/     \o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\     /o\     /o\     /o\     /o\
      /   \   /   \   /   \   /   \   /   \
    o/     \o/     \o/     \o/     \o/     \o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  |
    |   8   |   4   |   3   |       |  12   |   3
    |       |       |       |       |       |
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


@pytest.mark.parametrize("action, build", _ROBBER_ACTIONS)
def test_no_victim(action: RobberStep, build: Callable[[], Board]) -> None:
    # A tile with no opponent buildings: move the robber, steal from no one.
    board = build()
    # Clear the opponent settlement so no one is robbable on tile 0.
    layout, st = board
    board = (layout, st._replace(vertex_owner=jnp.zeros_like(st.vertex_owner)))
    before = np.asarray(board[1].player_resources)
    state, result = action(board, (jnp.array([0]), jnp.array([-1])))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.robber[0]) == 0
    assert np.array_equal(np.asarray(state.player_resources), before)


@pytest.mark.parametrize("action, build", _ROBBER_ACTIONS)
def test_invalid_tile_is_robber(action: RobberStep, build: Callable[[], Board]) -> None:
    board = set_robber(build(), 0)  # robber already on tile 0
    before = np.asarray(board[1].player_resources)
    state, result = action(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


@pytest.mark.parametrize("action, build", _ROBBER_ACTIONS)
def test_invalid_out_of_range_tile(
    action: RobberStep, build: Callable[[], Board]
) -> None:
    board = build()
    before = np.asarray(board[1].player_resources)
    state, result = action(board, (jnp.array([999]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_wrong_phase(robber_board: Board) -> None:
    board = set_phase(robber_board, GamePhase.MAIN)  # not MOVE_ROBBER
    before = np.asarray(board[1].player_resources)
    state, result = move_robber_step(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)
