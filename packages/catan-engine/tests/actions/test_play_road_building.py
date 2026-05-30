"""Tests for the vectorized PlayRoadBuilding action."""

from expecttest import assert_expected_inline

from catan_engine.action import ActionResult, PlayRoadBuilding
from catan_engine.board import Board, make_board, set_phase, to_main
from catan_engine.dev_cards import DevCard
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def test_success(road_building_board: Board) -> None:
    state, result = PlayRoadBuilding()(road_building_board, None)
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


def test_invalid_wrong_phase(road_building_board: Board) -> None:
    board = set_phase(road_building_board, GamePhase.ROLL)
    _, result = PlayRoadBuilding()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value


def test_invalid_no_card() -> None:
    board = to_main(make_board())  # no Road Building card granted
    _, result = PlayRoadBuilding()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value


def test_invalid_dev_already_played(road_building_board: Board) -> None:
    layout, st = road_building_board
    st = st._replace(dev_played=st.dev_played.at[0].set(1))
    _, result = PlayRoadBuilding()((layout, st), None)
    assert int(result[0]) == ActionResult.INVALID.value
