"""Tests for the vectorized PlayKnight action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, PlayKnight
from catan_engine.board import (
    Board,
    give_dev_card,
    make_board,
    place_settlement,
    set_robber,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.layout import TILE_V
from tests.mechanics.actions.fixtures import fmt

_TILE_V = np.asarray(TILE_V)


def test_success(knight_board: Board, render: Callable[..., str]) -> None:
    # Robber starts on tile 1; play a knight to tile 0 and steal from player 1.
    state, result = PlayKnight()(knight_board, (jnp.array([0]), jnp.array([1])))
    assert_expected_inline(
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
    assert_expected_inline(render(knight_board[0], state), """\



          ORE             3:1
               /o\\     /o\\     /o\\
              /   \\   /   \\   /   \\
            o/     \\o/     \\o/     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |  <R>  |
           /o\\     /o\\     /o\\     /o\\   3:1
          /   \\   /   \\   /   \\   /   \\
        o/     \\o/     \\o/     \\2/     \\o
  WOD   |  WHT  |  WOD  |  WOD  |  SHP  |
        |   9   |   2   |  10   |  11   |
        |       |       |       |       |
       /o\\     /o\\     /o\\     /o\\     /o\\
      /   \\   /   \\   /   \\   /   \\   /   \\
    o/     \\o/     \\o/     \\o/     \\o/     \\o
    |  ORE  |  SHP  |  WOD  |  DST  |  WHT  |
    |   8   |   4   |   3   |       |  12   |   3
    |       |       |       |       |       |
    o\\     /o\\     /o\\     /o\\     /o\\     /o
      \\   /   \\   /   \\   /   \\   /   \\   /
       \\o/     \\o/     \\o/     \\o/     \\o/
        |  SHP  |  ORE  |  BRK  |  BRK  |
        |   8   |   3   |  11   |   6   |
  3:1   |       |       |       |       |
        o\\     /o\\     /o\\     /o\\     /o
          \\   /   \\   /   \\   /   \\   /
           \\o/     \\o/     \\o/     \\o/   BRK
            |  WHT  |  WHT  |  WOD  |
            |   4   |   9   |   5   |
            |       |       |       |
            o\\     /o\\     /o\\     /o
              \\   /   \\   /   \\   /
               \\o/     \\o/     \\o/
          SHP             WHT


""")


def test_no_victim() -> None:
    # A tile with no opponent buildings: move the robber, steal from no one.
    board = to_main(make_board(seed=0))
    board = give_dev_card(board, 0, DevCard.KNIGHT)
    board = set_robber(board, 1 % _TILE_V.shape[0])
    before = np.asarray(board[1].player_resources)
    state, result = PlayKnight()(board, (jnp.array([0]), jnp.array([-1])))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.robber[0]) == 0
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_no_knight() -> None:
    board = to_main(make_board(seed=0))
    board = place_settlement(board, 1, int(_TILE_V[0, 0]))
    board = set_robber(board, 1 % _TILE_V.shape[0])
    before = np.asarray(board[1].player_resources)
    state, result = PlayKnight()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_tile_is_robber(knight_board: Board) -> None:
    board = set_robber(knight_board, 0)  # robber already on tile 0
    before = np.asarray(board[1].player_resources)
    state, result = PlayKnight()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_out_of_range_tile(knight_board: Board) -> None:
    _, result = PlayKnight()(knight_board, (jnp.array([999]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value


def test_invalid_dev_already_played(knight_board: Board) -> None:
    layout, st = knight_board
    board = (layout, st._replace(dev_played=st.dev_played.at[0].set(1)))
    before = np.asarray(board[1].player_resources)
    state, result = PlayKnight()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)
