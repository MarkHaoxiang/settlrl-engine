"""Tests for the vectorized PlayRoadBuilding action."""

import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import PlayRoadBuilding
from catan_engine.board import give_dev_card, make_board, set_phase, to_main
from catan_engine.dev_cards import DevCard
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _ready() -> tuple:
    """A MAIN-phase board where player 0 holds a Road Building card."""
    board = to_main(make_board())
    return give_dev_card(board, 0, DevCard.ROAD_BUILDING)


class TestPlayRoadBuilding(TestCase):
    def test_success(self) -> None:
        board = _ready()
        state, result = PlayRoadBuilding()(board, None)
        self.assertExpectedInline(
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

    def test_invalid_wrong_phase(self) -> None:
        board = set_phase(_ready(), GamePhase.ROLL)
        _, result = PlayRoadBuilding()(board, None)
        assert int(result[0]) == ActionResult.INVALID.value

    def test_invalid_no_card(self) -> None:
        board = to_main(make_board())  # no Road Building card granted
        _, result = PlayRoadBuilding()(board, None)
        assert int(result[0]) == ActionResult.INVALID.value

    def test_invalid_dev_already_played(self) -> None:
        layout, st = _ready()
        st = st._replace(dev_played=st.dev_played.at[0].set(1))
        _, result = PlayRoadBuilding()((layout, st), None)
        assert int(result[0]) == ActionResult.INVALID.value
