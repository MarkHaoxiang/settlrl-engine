"""Tests for the vectorized BuyDevelopmentCard action."""

import numpy as np
from catan_engine.board import Board
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.development import buy_development_card_step
from expecttest import assert_expected_inline

from tests.mechanics.actions.fixtures import fmt


def test_success(buy_board: Board) -> None:
    state, result = buy_development_card_step(buy_board, None)
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


# Wrong-phase and cannot-afford rejections are covered by the parametrized
# tests in test_invalid_paths.py.


def test_invalid_empty_deck(buy_board: Board) -> None:
    layout, st = buy_board
    st = st._replace(dev_deck=st.dev_deck.at[0].set(0))
    _, result = buy_development_card_step((layout, st), None)
    assert int(result[0]) == ActionResult.INVALID.value
