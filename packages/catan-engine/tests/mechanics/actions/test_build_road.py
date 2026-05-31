"""Tests for the vectorized BuildRoad action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.development import play_road_building_step
from catan_engine.mechanics.placement import build_road_step
from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    place_road,
    place_settlement,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from tests.mechanics.actions.fixtures import edge_path_from, fmt


def test_success(road_board: tuple[Board, int], render: Callable[..., str]) -> None:
    board, edge = road_board
    state, result = build_road_step(board, jnp.array([edge]))
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
    assert_expected_inline(
        render(board[0], state),
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


# Wrong-phase and cannot-afford rejections are covered by the parametrized
# tests in test_invalid_paths.py.


def test_invalid_out_of_range(road_board: tuple[Board, int]) -> None:
    board, _ = road_board
    before = np.asarray(board[1].edge_road)
    state, result = build_road_step(board, jnp.array([-1]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.edge_road), before)


def test_win_via_longest_road() -> None:
    # Player 0 at 8 VP (1 settlement + 7 hidden VP cards); a 5th road flips
    # Longest Road (+2) to cross 10. Build a 4-road chain, then the 5th edge.
    path = edge_path_from(0, 5)
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    for e in path[:4]:
        board = place_road(board, 0, e)
    board = give(board, 0, [0, 0, 1, 1, 0])  # one road's worth
    board = give_dev_card(board, 0, DevCard.VICTORY_POINT, 7)
    state, result = build_road_step(board, jnp.array([path[4]]))
    assert int(result[0]) == ActionResult.GAME_COMPLETE.value
    assert int(state.longest_road_owner[0]) == 0
    assert int(state.longest_road_len[0]) == 5


def test_invalid_road_stock_exhausted() -> None:
    # Place all 15 roads along a chain, fund a 16th, and confirm the only reason
    # the otherwise-connected 16th edge is rejected is the MAX_ROADS cap.
    path = edge_path_from(0, 16)
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    for e in path[:15]:
        board = place_road(board, 0, e)
    board = give(board, 0, [0, 0, 5, 5, 0])  # plenty of wood + brick
    _, result = build_road_step(board, jnp.array([path[15]]))
    assert int(result[0]) == ActionResult.INVALID.value


def test_road_building_free_road_chain() -> None:
    # Road Building grants free_roads=2; with ZERO resources two roads succeed
    # (decrementing 2 -> 1 -> 0) and a third (no free roads, no resources) fails.
    path = edge_path_from(0, 3)
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    board = give(board, 0, [0, 0, 0, 0, 0])  # no resources at all
    board = give_dev_card(board, 0, DevCard.ROAD_BUILDING)

    state, result = play_road_building_step(board, None)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.free_roads[0]) == 2

    state, result = build_road_step((board[0], state), jnp.array([path[0]]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.free_roads[0]) == 1

    state, result = build_road_step((board[0], state), jnp.array([path[1]]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.free_roads[0]) == 0

    _, result = build_road_step((board[0], state), jnp.array([path[2]]))
    assert int(result[0]) == ActionResult.INVALID.value
