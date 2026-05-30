"""Tests for the vectorized BuildRoad action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.action import ActionResult, BuildRoad
from catan_engine.board import Board, give, set_phase
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def test_success(road_board: tuple[Board, int], render: Callable[..., str]) -> None:
    board, edge = road_board
    state, result = BuildRoad()(board, jnp.array([edge]))
    assert_expected_inline(
        fmt(
            result,
            edge_owner=int(state.edge_road[0, edge]),
            wood=int(state.player_resources[0, 0, 2]),
            brick=int(state.player_resources[0, 0, 3]),
            roads=int((np.asarray(state.edge_road[0]) == 1).sum()),
        ),
        """\
result=OK
edge_owner=1
wood=0
brick=0
roads=1""",
    )
    assert_expected_inline(render(board[0], state), """\



          ORE             3:1
               /o\\     /o\\     /o\\
              /   \\   /   \\   /   \\
            o/     \\o/     \\o/     \\o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |  <R>  |
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


def test_invalid_wrong_phase(road_board: tuple[Board, int]) -> None:
    board, edge = road_board
    board = set_phase(board, GamePhase.ROLL)
    before = np.asarray(board[1].edge_road)
    state, result = BuildRoad()(board, jnp.array([edge]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.edge_road), before)


def test_invalid_out_of_range(road_board: tuple[Board, int]) -> None:
    board, _ = road_board
    before = np.asarray(board[1].edge_road)
    state, result = BuildRoad()(board, jnp.array([-1]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.edge_road), before)


def test_invalid_cannot_afford(road_board: tuple[Board, int]) -> None:
    board, edge = road_board
    board = give(board, 0, [0, 0, 0, 0, 0])  # no wood/brick, no free roads
    _, result = BuildRoad()(board, jnp.array([edge]))
    assert int(result[0]) == ActionResult.INVALID.value
