"""Tests for the vectorized SetupSettlement action."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline
from settlrl_engine.board import Board, place_settlement, set_phase
from settlrl_engine.board.layout import EDGE_V
from settlrl_engine.board.resources import N_PLAYERS
from settlrl_engine.board.state import GamePhase
from settlrl_engine.mechanics.action import ActionResult
from settlrl_engine.mechanics.setup import setup_settlement_step

from tests.mechanics.actions.fixtures import fmt

# A vertex sharing an edge with vertex 0 (for the distance-rule test).
_NBR_OF_0 = int(
    next(b if a == 0 else a for a, b in np.asarray(EDGE_V).tolist() if 0 in (a, b))
)


def test_success(setup_board: Board, render: Callable[..., str]) -> None:
    state, result = setup_settlement_step(setup_board, jnp.array([0]))
    assert_expected_inline(
        fmt(
            result,
            owner=int(state.vertex_owner[0, 0]),
            kind=int(state.vertex_type[0, 0]),
            vp=int(state.victory_points[0, 0]),
            phase=str(GamePhase(int(state.phase[0]))),
            resources_total=int(np.asarray(state.player_resources[0, 0]).sum()),
        ),
        """\
result=OK
owner=1
kind=1
vp=1
phase=SETUP_ROAD
resources_total=0""",
    )
    assert_expected_inline(
        render(setup_board[0], state),
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
        o/     \o/     \o/     \1/     \o
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


def test_second_settlement_grants_resources(setup_board: Board) -> None:
    layout, st = setup_board
    st = st._replace(setup_index=st.setup_index.at[0].set(N_PLAYERS))
    state, result = setup_settlement_step((layout, st), jnp.array([0]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(np.asarray(state.player_resources[0, 0]).sum()) > 0


def test_invalid_wrong_phase(setup_board: Board) -> None:
    board = set_phase(setup_board, GamePhase.ROLL)
    before = np.asarray(board[1].vertex_owner)
    state, result = setup_settlement_step(board, jnp.array([0]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.vertex_owner), before)


def test_invalid_distance_rule(setup_board: Board) -> None:
    board = place_settlement(setup_board, 0, 0)
    _, result = setup_settlement_step(board, jnp.array([_NBR_OF_0]))
    assert int(result[0]) == ActionResult.INVALID.value
