"""Tests for the vectorized BuildCity action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, BuildCity
from catan_engine.board import Board, give, set_phase
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt


def test_success(city_board: tuple[Board, int], render: Callable[..., str]) -> None:
    board, vertex = city_board
    state, result = BuildCity()(board, jnp.array([vertex]))
    assert_expected_inline(
        fmt(
            result,
            kind=int(state.vertex_type[0, vertex]),
            vp=int(state.victory_points[0, 0]),
            wheat=int(state.player_resources[0, 0, 1]),
            ore=int(state.player_resources[0, 0, 4]),
        ),
        """\
result=OK
kind=2
vp=2
wheat=0
ore=0""",
    )
    assert_expected_inline(render(board[0], state), """\



          ORE             3:1
               /o\\     /o\\     /o\\
              /   \\   /   \\   /   \\
            o/     \\o/     \\o/     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |  <R>  |
           /o\\     /o\\     /o\\     /o\\   3:1
          /   \\   /   \\   /   \\   /   \\
        o/     \\o/     \\o/     \\A/     \\o
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


def test_invalid_wrong_phase(city_board: tuple[Board, int]) -> None:
    board, vertex = city_board
    board = set_phase(board, GamePhase.ROLL)
    before = np.asarray(board[1].vertex_type)
    state, result = BuildCity()(board, jnp.array([vertex]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.vertex_type), before)


def test_invalid_no_own_settlement(city_board: tuple[Board, int]) -> None:
    # A distant empty vertex holds no settlement of the player's.
    board, _ = city_board
    lonely = 40
    before = np.asarray(board[1].vertex_type)
    state, result = BuildCity()(board, jnp.array([lonely]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.vertex_type), before)


def test_invalid_cannot_afford(city_board: tuple[Board, int]) -> None:
    board, vertex = city_board
    board = give(board, 0, [0, 0, 0, 0, 0])
    _, result = BuildCity()(board, jnp.array([vertex]))
    assert int(result[0]) == ActionResult.INVALID.value
