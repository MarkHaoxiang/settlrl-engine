"""Tests for the vectorized BuyDevelopmentCard action."""

import numpy as np
from expecttest import assert_expected_inline
from settlrl_engine.board import Board
from settlrl_engine.mechanics.action import ActionResult
from settlrl_engine.mechanics.development import buy_development_card_step

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


def test_forced_card_type(buy_board: Board) -> None:
    """params=t+1 forces card type t: deck decrements there, hand gains it."""
    before_deck = np.asarray(buy_board[1].dev_deck[0])
    state, result = buy_development_card_step(buy_board, 2)  # force type 1
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.dev_hand[0, 0, 1]) == 1
    assert int(state.dev_deck[0, 1]) == before_deck[1] - 1


def test_third_party_at_threshold_does_not_complete(buy_board: Board) -> None:
    # Only the current player can win (rulebook p.5): an opponent already
    # sitting at 10 VP (crowned by a settlement break on an earlier turn)
    # leaves the buyer's successful action SUCCESS, and play continues.
    layout, st = buy_board
    st = st._replace(victory_points=st.victory_points.at[0, 1].set(10))
    _, result = buy_development_card_step((layout, st), 2)  # a non-VP card
    assert int(result[0]) == ActionResult.SUCCESS.value


def test_forced_out_of_stock_is_invalid(buy_board: Board) -> None:
    layout, st = buy_board
    st = st._replace(dev_deck=st.dev_deck.at[0, 1].set(0))
    state, result = buy_development_card_step((layout, st), 2)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.dev_hand), np.asarray(st.dev_hand))
