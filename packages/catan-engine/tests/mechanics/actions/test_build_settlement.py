"""Tests for the vectorized BuildSettlement action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, BuildSettlement
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
from tests.mechanics.actions.fixtures import (
    edge_path_from,
    fmt,
    independent_vertices,
    settlement_fixture,
)


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
    assert_expected_inline(
        render(board[0], state),
        r"""


          ORE             3:1
               /o\     /o\     /o\
              /   \   /   \   /   \
            o/     \o/     \o/     \1
            |  SHP  |  ORE  |  BRK  1
            |   5   |   6   |  10   1
            |       |       |       1
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


def test_invalid_not_connected(settlement_board: tuple[Board, int]) -> None:
    # A distant empty vertex with no adjacent road is not connected.
    board, _ = settlement_board
    lonely = 40
    _, result = BuildSettlement()(board, jnp.array([lonely]))
    assert int(result[0]) == ActionResult.INVALID.value


def test_win_crossing_ten_vp() -> None:
    # Player 0 holds 8 hidden VP cards + a settlement (1 building VP) = 9 VP;
    # the new settlement's +1 building VP crosses to 10.
    board, vertex = settlement_fixture()
    board = give_dev_card(board, 0, DevCard.VICTORY_POINT, 8)
    state, result = BuildSettlement()(board, jnp.array([vertex]))
    assert int(result[0]) == ActionResult.GAME_COMPLETE.value
    assert int(state.victory_points[0, 0]) == 2  # two on-board settlements


def test_invalid_settlement_stock_exhausted() -> None:
    # Place all 5 settlements (one at the spur root), fund and connect a 6th
    # legal vertex, and confirm only the MAX_SETTLEMENTS cap rejects it.
    from catan_engine.board.layout import EDGE_V

    edge_v = np.asarray(EDGE_V)
    v0 = 0
    # Build a 2-edge spur off v0 to a connected, distance-legal target x.
    e0, e1 = edge_path_from(v0, 2)
    a, b = int(edge_v[e0, 0]), int(edge_v[e0, 1])
    w = b if a == v0 else a
    a2, b2 = int(edge_v[e1, 0]), int(edge_v[e1, 1])
    x = b2 if a2 == w else a2

    # Choose 5 settlement sites: v0 plus 4 more, none adjacent to the spur/x.
    forbidden = {w, x}
    sites = [v0]
    for v in independent_vertices(12):
        if len(sites) == 5:
            break
        if v == v0 or v in forbidden:
            continue
        sites.append(v)
    board = to_main(make_board())
    for v in sites:
        board = place_settlement(board, 0, v)
    board = place_road(board, 0, e0)
    board = place_road(board, 0, e1)
    board = give(board, 0, [1, 1, 1, 1, 0])  # one settlement's worth
    _, result = BuildSettlement()(board, jnp.array([x]))
    assert int(result[0]) == ActionResult.INVALID.value
