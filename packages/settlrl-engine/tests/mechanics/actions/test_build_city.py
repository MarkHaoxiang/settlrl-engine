"""Tests for the vectorized BuildCity action."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline
from settlrl_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    place_city,
    place_settlement,
    to_main,
)
from settlrl_engine.board.dev_cards import DevCard
from settlrl_engine.mechanics.action import ActionResult
from settlrl_engine.mechanics.placement import build_city_step

from tests.mechanics.actions.fixtures import fmt, independent_vertices


def test_success(city_board: tuple[Board, int], render: Callable[..., str]) -> None:
    board, vertex = city_board
    state, result = build_city_step(board, jnp.array([vertex]))
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
    assert_expected_inline(
        render(board[0], state),
        r"""


          BRK             SHP
               /o\     /o\     /o\
              /   \   /   \   /   \
            o/     \o/     \o/     \o
            |  WOD  |  WHT  |  ORE  |
            |   8   |   6   |   8   |
            |       |       |       |
           /o\     /o\     /o\     /o\   3:1
          /   \   /   \   /   \   /   \
        o/     \o/     \o/     \A/     \o
  ORE   |  WHT  |  BRK  |  SHP  |  SHP  |
        |  10   |   5   |  12   |   4   |
        |       |       |       |       |
       /o\     /o\     /o\     /o\     /o\
      /   \   /   \   /   \   /   \   /   \
    o/     \o/     \o/     \o/     \o/     \o
    |  WHT  |  ORE  |  SHP  |  SHP  |  WOD  |
    |   5   |   3   |   9   |   6   |  11   |   3
    |       |       |       |       |       |
    o\     /o\     /o\     /o\     /o\     /o
      \   /   \   /   \   /   \   /   \   /
       \o/     \o/     \o/     \o/     \o/
        |  BRK  |  WOD  |  WHT  |  WOD  |
        |   3   |   2   |   4   |  10   |
  3:1   |       |       |       |       |
        o\     /o\     /o\     /o\     /o
          \   /   \   /   \   /   \   /
           \o/     \o/     \o/     \o/   3:1
            |  DST  |  ORE  |  BRK  |
            |       |   9   |  11   |
            |  <R>  |       |       |
            o\     /o\     /o\     /o
              \   /   \   /   \   /
               \o/     \o/     \o/
          WHT             WOD


""",
    )


# Wrong-phase and cannot-afford rejections are covered by the parametrized
# tests in test_invalid_paths.py.


def test_invalid_no_own_settlement(city_board: tuple[Board, int]) -> None:
    # A distant empty vertex holds no settlement of the player's.
    board, _ = city_board
    lonely = 40
    before = np.asarray(board[1].vertex_type)
    state, result = build_city_step(board, jnp.array([lonely]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.vertex_type), before)


def test_win_crossing_ten_vp(city_board: tuple[Board, int]) -> None:
    # Player 0: settlement (1 building VP) + 8 hidden VP cards = 9 VP; the city
    # upgrade's +1 building VP crosses to 10.
    board, vertex = city_board
    board = give_dev_card(board, 0, DevCard.VICTORY_POINT, 8)
    state, result = build_city_step(board, jnp.array([vertex]))
    assert int(result[0]) == ActionResult.GAME_COMPLETE.value
    assert int(state.victory_points[0, 0]) == 2  # settlement(1) + city upgrade(+1)


def test_invalid_city_stock_exhausted() -> None:
    # Place all 4 cities plus a 5th (upgradeable) settlement, fund the upgrade,
    # and confirm only the MAX_CITIES cap rejects it.
    sites = independent_vertices(5)
    board = to_main(make_board())
    for v in sites[:4]:
        board = place_city(board, 0, v)
    board = place_settlement(board, 0, sites[4])
    board = give(board, 0, [0, 2, 0, 0, 3])  # one city's worth
    _, result = build_city_step(board, jnp.array([sites[4]]))
    assert int(result[0]) == ActionResult.INVALID.value
