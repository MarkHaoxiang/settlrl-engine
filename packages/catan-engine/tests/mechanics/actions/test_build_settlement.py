"""Tests for the vectorized BuildSettlement action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, BuildSettlement
from catan_engine.board import Board, give, set_phase
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt


def test_success(
    settlement_board: tuple[Board, int], render: Callable[..., str]
) -> None:
    board, vertex = settlement_board
    state, result = BuildSettlement()(board, jnp.array([vertex]))
    assert_expected_inline(
        fmt(
            result,
            owner=int(state.vertex_owner[0, vertex]),
            kind=int(state.vertex_type[0, vertex]),
            vp=int(state.victory_points[0, 0]),
            resources=int(np.asarray(state.player_resources[0, 0]).sum()),
        ),
        """\
result=OK
owner=1
kind=1
vp=2
resources=0""",
    )
    assert_expected_inline(render(board[0], state), """\



          ORE             3:1
               /o\\     /o\\     /o\\
              /   \\   /   \\   /   \\
            o/     \\o/     \\o/     \\1
            |  SHP  |  ORE  |  BRK  1
            |   5   |   6   |  10   1
            |       |       |  <R>  1
           /o\\     /o\\     /o\\     1o\\   3:1
          /   \\   /   \\   /   \\   1   \\
        o/     \\o/     \\o/     \\11     \\o
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


def test_invalid_wrong_phase(settlement_board: tuple[Board, int]) -> None:
    board, vertex = settlement_board
    board = set_phase(board, GamePhase.ROLL)
    before = np.asarray(board[1].vertex_owner)
    state, result = BuildSettlement()(board, jnp.array([vertex]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.vertex_owner), before)


def test_invalid_not_connected(settlement_board: tuple[Board, int]) -> None:
    # A distant empty vertex with no adjacent road is not connected.
    board, _ = settlement_board
    lonely = 40
    _, result = BuildSettlement()(board, jnp.array([lonely]))
    assert int(result[0]) == ActionResult.INVALID.value


def test_invalid_cannot_afford(settlement_board: tuple[Board, int]) -> None:
    board, vertex = settlement_board
    board = give(board, 0, [0, 0, 0, 0, 0])
    _, result = BuildSettlement()(board, jnp.array([vertex]))
    assert int(result[0]) == ActionResult.INVALID.value
