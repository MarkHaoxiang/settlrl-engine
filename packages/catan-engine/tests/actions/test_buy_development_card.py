"""Tests for the vectorized BuyDevelopmentCard action."""

import numpy as np
from expecttest import assert_expected_inline

from catan_engine.action import ActionResult, BuyDevelopmentCard
from catan_engine.board import Board, give, set_phase
from catan_engine.state import GamePhase
from tests.actions.fixtures import fmt


def test_success(buy_board: Board) -> None:
    state, result = BuyDevelopmentCard()(buy_board, None)
    hand = np.asarray(state.dev_hand[0, 0])
    assert_expected_inline(
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


def test_invalid_wrong_phase(buy_board: Board) -> None:
    board = set_phase(buy_board, GamePhase.ROLL)
    before = np.asarray(board[1].dev_deck)
    state, result = BuyDevelopmentCard()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.dev_deck), before)


def test_invalid_cannot_afford(buy_board: Board) -> None:
    board = give(buy_board, 0, [0, 0, 0, 0, 0])
    _, result = BuyDevelopmentCard()(board, None)
    assert int(result[0]) == ActionResult.INVALID.value


def test_invalid_empty_deck(buy_board: Board) -> None:
    layout, st = buy_board
    st = st._replace(dev_deck=st.dev_deck.at[0].set(0))
    _, result = BuyDevelopmentCard()((layout, st), None)
    assert int(result[0]) == ActionResult.INVALID.value
