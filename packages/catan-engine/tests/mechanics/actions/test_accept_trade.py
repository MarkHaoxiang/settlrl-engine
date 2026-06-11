"""Tests for the vectorized AcceptTrade action."""

import numpy as np
from catan_engine.board import Board
from catan_engine.board.state import NO_INDEX, GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.trade import accept_trade_step
from expecttest import assert_expected_inline

from tests.mechanics.actions.fixtures import fmt

SHEEP, WOOD = 0, 2


def test_accept_swaps_the_cards(response_board: Board) -> None:
    # Pending offer: player 0 gives a sheep to player 2 for a wood.
    state, result = accept_trade_step(response_board)
    assert_expected_inline(
        fmt(
            result,
            phase=str(GamePhase(int(state.phase[0]))),
            proposer=np.asarray(state.player_resources[0, 0]).tolist(),
            partner=np.asarray(state.player_resources[0, 2]).tolist(),
            cleared=int(state.trade_partner[0]) == NO_INDEX,
        ),
        """\
result=OK
phase=MAIN
proposer=[0, 0, 1, 0, 0]
partner=[1, 0, 2, 0, 0]
cleared=True""",
    )


def test_invalid_partner_lacks_asked_card(response_board: Board) -> None:
    # Strip player 2's wood after the offer: only Reject remains legal.
    layout, st = response_board
    st = st._replace(player_resources=st.player_resources.at[0, 2, WOOD].set(0))
    state, result = accept_trade_step((layout, st))
    assert int(result[0]) == ActionResult.INVALID.value
    assert int(state.phase[0]) == GamePhase.TRADE_RESPONSE


def test_invalid_wrong_phase(propose_board: Board) -> None:
    before = np.asarray(propose_board[1].player_resources)
    state, result = accept_trade_step(propose_board)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)
