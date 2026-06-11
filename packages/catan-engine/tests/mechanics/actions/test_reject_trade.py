"""Tests for the vectorized RejectTrade action."""

import numpy as np
from catan_engine.board import Board
from catan_engine.board.state import NO_INDEX, GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.trade import reject_trade_step


def test_reject_clears_the_offer_unchanged(response_board: Board) -> None:
    state, result = reject_trade_step(response_board)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.phase[0]) == GamePhase.MAIN
    assert int(state.trade_partner[0]) == NO_INDEX
    assert np.array_equal(
        np.asarray(state.player_resources),
        np.asarray(response_board[1].player_resources),
    )


def test_invalid_wrong_phase(propose_board: Board) -> None:
    state, result = reject_trade_step(propose_board)
    assert int(result[0]) == ActionResult.INVALID.value
    assert int(state.phase[0]) == GamePhase.MAIN
