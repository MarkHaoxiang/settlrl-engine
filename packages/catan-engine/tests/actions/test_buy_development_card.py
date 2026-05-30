"""Tests for the vectorized BuyDevelopmentCard action."""

import numpy as np
from expecttest import TestCase

from catan_engine.action_vec import ActionResult
from catan_engine.action_vec import BuyDevelopmentCard
from catan_engine.board import Board, give, make_board, set_phase, to_main
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def _buy_ready() -> Board:
    """MAIN board (fixed seed) where player 0 can afford exactly one dev card."""
    board = to_main(make_board(seed=0))
    return give(board, 0, [1, 1, 0, 0, 1])  # sheep + wheat + ore


class TestBuyDevelopmentCard(TestCase):
    def test_success(self) -> None:
        board = _buy_ready()
        state, result = BuyDevelopmentCard()(board, None)
        hand = np.asarray(state.dev_hand[0, 0])
        self.assertExpectedInline(
            fmt(
                result,
                card=int(hand.argmax()),
                hand_total=int(hand.sum()),
                resources_total=int(np.asarray(state.player_resources[0, 0]).sum()),
                deck_total=int(np.asarray(state.dev_deck[0]).sum()),
            ),
            """\
result=OK
card=4
hand_total=1
resources_total=0
deck_total=24""",
        )

    def test_invalid_wrong_phase(self) -> None:
        board = set_phase(_buy_ready(), GamePhase.ROLL)
        before = np.asarray(board[1].dev_deck)
        state, result = BuyDevelopmentCard()(board, None)
        assert int(result[0]) == ActionResult.INVALID.value
        assert np.array_equal(np.asarray(state.dev_deck), before)

    def test_invalid_cannot_afford(self) -> None:
        board = give(_buy_ready(), 0, [0, 0, 0, 0, 0])
        _, result = BuyDevelopmentCard()(board, None)
        assert int(result[0]) == ActionResult.INVALID.value

    def test_invalid_empty_deck(self) -> None:
        layout, st = _buy_ready()
        st = st._replace(dev_deck=st.dev_deck.at[0].set(0))
        _, result = BuyDevelopmentCard()((layout, st), None)
        assert int(result[0]) == ActionResult.INVALID.value
