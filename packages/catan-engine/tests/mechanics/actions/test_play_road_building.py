"""Tests for the vectorized PlayRoadBuilding action."""

import jax.numpy as jnp
from catan_engine.board import (
    Board,
    give_dev_card,
    make_board,
    place_settlement,
    set_phase,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.state import GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.development import play_road_building_step
from catan_engine.mechanics.dice import roll_step
from catan_engine.mechanics.placement import build_road_step
from expecttest import assert_expected_inline

from tests.mechanics.actions.fixtures import first_legal_edge, fmt


def test_success(road_building_board: Board) -> None:
    state, result = play_road_building_step(road_building_board, None)
    assert_expected_inline(
        fmt(
            result,
            free_roads=int(state.free_roads[0]),
            dev_played=int(state.dev_played[0]),
            hand=int(state.dev_hand[0, 0, DevCard.ROAD_BUILDING]),
        ),
        """\
result=OK
free_roads=2
dev_played=1
hand=0""",
    )


# Wrong-phase rejection is covered by the parametrized test in
# test_invalid_paths.py.


def test_invalid_no_card() -> None:
    board = to_main(make_board())  # no Road Building card granted
    _, result = play_road_building_step(board, None)
    assert int(result[0]) == ActionResult.INVALID.value


def test_invalid_dev_already_played(road_building_board: Board) -> None:
    layout, st = road_building_board
    st = st._replace(dev_played=st.dev_played.at[0].set(1))
    _, result = play_road_building_step((layout, st), None)
    assert int(result[0]) == ActionResult.INVALID.value


def test_playable_before_the_roll_and_roads_place_pre_roll() -> None:
    # Rulebook: the one dev card may be played any time during the turn, even
    # before rolling — and Road Building's free roads place immediately.
    board = set_phase(make_board(seed=0), GamePhase.ROLL)
    board = place_settlement(board, 0, 0)
    board = give_dev_card(board, 0, DevCard.ROAD_BUILDING)

    state, result = play_road_building_step(board, None)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.free_roads[0]) == 2
    assert int(state.phase[0]) == GamePhase.ROLL  # the roll is still owed

    board = (board[0], state)
    state, result = build_road_step(board, jnp.array([first_legal_edge(board)]))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.free_roads[0]) == 1

    # Rolling stays legal mid-placement (free roads may be deferred to MAIN).
    _, result = roll_step((board[0], state))
    assert int(result[0]) == ActionResult.SUCCESS.value
