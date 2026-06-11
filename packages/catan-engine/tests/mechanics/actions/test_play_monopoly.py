"""Tests for the vectorized PlayMonopoly action."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    set_phase,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.state import GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.development import play_monopoly_step
from expecttest import assert_expected_inline

from tests.mechanics.actions.fixtures import fmt


def test_success(monopoly_board: Board, render: Callable[..., str]) -> None:
    state, result = play_monopoly_step(monopoly_board, jnp.array([0]))
    assert_expected_inline(
        fmt(
            result,
            player0_sheep=int(state.player_resources[0, 0, 0]),
            player1_sheep=int(state.player_resources[0, 1, 0]),
            player2_sheep=int(state.player_resources[0, 2, 0]),
            dev_played=int(state.dev_played[0]),
            player0_monopoly=int(state.dev_hand[0, 0, DevCard.MONOPOLY]),
        ),
        """\
result=OK
player0_sheep=6
player1_sheep=0
player2_sheep=0
dev_played=1
player0_monopoly=0""",
    )
    # The full board's Players table shows all sheep swept onto player 1.
    assert_expected_inline(
        render(monopoly_board[0], state, full=True),
        r"""Catan Board
============================================================




          ORE             3:1
               /o\     /o\     /o\
              /   \   /   \   /   \
            o/     \o/     \o/     \o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |       |
           /o\     /o\     /o\     /o\   3:1
          /   \   /   \   /   \   /   \
        o/     \o/     \o/     \o/     \o
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




Legend: o=vertex  1-4=settlement(player)  A-D=city(player)  digit on edge=road(player)  <R>=robber

Phase MAIN  |  Current player 1  |  Dice -

Players
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   P |   Shp |   Wht |   Wod |   Brk |   Ore |   Hand |   Dev |   Knt |   Set |   Cit |   Rd |   VP | Awd   |
+====+=====+=======+=======+=======+=======+=======+========+=======+=======+=======+=======+======+======+=======+
| >  |   1 |     6 |     0 |     0 |     0 |     0 |      6 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   2 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   3 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+
|    |   4 |     0 |     0 |     0 |     0 |     0 |      0 |     0 |     0 |     0 |     0 |    0 |    0 |       |
+----+-----+-------+-------+-------+-------+-------+--------+-------+-------+-------+-------+------+------+-------+

Longest Road: unclaimed    Largest Army: unclaimed

Dev deck  KNT:14  RDB:2  YOP:2  MNP:2  VPT:5

Bank
+---------+---------+--------+---------+-------+
|   Sheep |   Wheat |   Wood |   Brick |   Ore |
+=========+=========+========+=========+=======+
|      13 |      19 |     19 |      19 |    19 |
+---------+---------+--------+---------+-------+

Robber: tile 8 (DST, desert)
""",
    )


# Wrong-phase rejection is covered by the parametrized test in
# test_invalid_paths.py.


def test_invalid_no_card() -> None:
    board = to_main(make_board())
    board = give(board, 0, [1, 0, 0, 0, 0])
    before = np.asarray(board[1].player_resources)
    state, result = play_monopoly_step(board, jnp.array([0]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_dev_already_played(monopoly_board: Board) -> None:
    layout, st = monopoly_board
    board = (layout, st._replace(dev_played=st.dev_played.at[0].set(1)))
    before = np.asarray(board[1].player_resources)
    new_state, result = play_monopoly_step(board, jnp.array([0]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(new_state.player_resources), before)


def test_invalid_out_of_range(monopoly_board: Board) -> None:
    before = np.asarray(monopoly_board[1].player_resources)
    state, result = play_monopoly_step(monopoly_board, jnp.array([-1]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_playable_before_the_roll() -> None:
    # Rulebook: the one dev card may be played any time during the turn.
    board = set_phase(make_board(seed=0), GamePhase.ROLL)
    board = give_dev_card(board, 0, DevCard.MONOPOLY)
    board = give(board, 1, [3, 0, 0, 0, 0])
    state, result = play_monopoly_step(board, jnp.array([0]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.player_resources[0, 0, 0]) == 3
    assert int(state.phase[0]) == GamePhase.ROLL  # the roll is still owed
