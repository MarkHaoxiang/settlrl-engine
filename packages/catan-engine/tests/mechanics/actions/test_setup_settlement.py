"""Tests for the vectorized SetupSettlement action."""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, SetupSettlement
from catan_engine.board import Board, place_settlement, set_phase
from catan_engine.board.layout import EDGE_V
from catan_engine.board.resources import N_PLAYERS
from catan_engine.board.state import GamePhase
from tests.mechanics.actions.fixtures import fmt

# A vertex sharing an edge with vertex 0 (for the distance-rule test).
_NBR_OF_0 = int(
    next(b if a == 0 else a for a, b in np.asarray(EDGE_V).tolist() if 0 in (a, b))
)


def test_success(setup_board: Board, render: Callable[..., str]) -> None:
    state, result = SetupSettlement()(setup_board, jnp.array([0]))
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


          ORE             3:1
               /o\     /o\     /o\
              /   \   /   \   /   \
            o/     \o/     \o/     \o
            |  SHP  |  ORE  |  BRK  |
            |   5   |   6   |  10   |
            |       |       |       |
           /o\     /o\     /o\     /o\   3:1
          /   \   /   \   /   \   /   \
        o/     \o/     \o/     \1/     \o
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


def test_second_settlement_grants_resources(setup_board: Board) -> None:
    layout, st = setup_board
    st = st._replace(setup_index=st.setup_index.at[0].set(N_PLAYERS))
    state, result = SetupSettlement()((layout, st), jnp.array([0]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(np.asarray(state.player_resources[0, 0]).sum()) > 0


def test_invalid_wrong_phase(setup_board: Board) -> None:
    board = set_phase(setup_board, GamePhase.ROLL)
    before = np.asarray(board[1].vertex_owner)
    state, result = SetupSettlement()(board, jnp.array([0]))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.vertex_owner), before)


def test_invalid_distance_rule(setup_board: Board) -> None:
    board = place_settlement(setup_board, 0, 0)
    _, result = SetupSettlement()(board, jnp.array([_NBR_OF_0]))
    assert int(result[0]) == ActionResult.INVALID.value
