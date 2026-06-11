"""Tests for the vectorized ProposeTrade action (1:1 domestic trade offer)."""

import jax
import jax.numpy as jnp
import numpy as np
from catan_engine.board import Board, give, make_board, to_main
from catan_engine.board.state import GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.common import agent_selection_single
from catan_engine.mechanics.trade import (
    pack_trade,
    propose_trade_available,
    propose_trade_step,
)
from expecttest import assert_expected_inline

from tests.mechanics.actions.fixtures import fmt

SHEEP, WHEAT, WOOD = 0, 1, 2


def _params(
    give_r: int, receive_r: int, partner: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.array([pack_trade(give_r, receive_r)]), jnp.array([partner])


def test_propose_parks_game_in_trade_response(propose_board: Board) -> None:
    state, result = propose_trade_step(propose_board, _params(SHEEP, WOOD, 2))
    assert_expected_inline(
        fmt(
            result,
            phase=str(GamePhase(int(state.phase[0]))),
            partner=int(state.trade_partner[0]),
            give=int(state.trade_give[0]),
            receive=int(state.trade_receive[0]),
            acting=int(jax.vmap(agent_selection_single)(state)[0]),
        ),
        """\
result=OK
phase=TRADE_RESPONSE
partner=2
give=0
receive=2
acting=2""",
    )
    # The offer itself moves no cards.
    assert np.array_equal(
        np.asarray(state.player_resources),
        np.asarray(propose_board[1].player_resources),
    )


def test_propose_ignores_partner_exact_holdings(propose_board: Board) -> None:
    # Player 2 holds no ore, but proposing sheep -> ore is still legal: only
    # public information (a non-empty hand) gates the offer.
    assert bool(propose_trade_available(propose_board, _params(SHEEP, 4, 2))[0])


def test_invalid_two_player_board() -> None:
    board = to_main(make_board(seed=0, n_players=2))
    board = give(board, 0, [1, 0, 0, 0, 0])
    board = give(board, 1, [0, 0, 3, 0, 0])
    state, result = propose_trade_step(board, _params(SHEEP, WOOD, 1))
    assert int(result[0]) == ActionResult.INVALID.value
    assert int(state.phase[0]) == GamePhase.MAIN


def test_invalid_partner_choices(propose_board: Board) -> None:
    for partner in (0, -1, 4):  # self / out of range
        assert not bool(
            propose_trade_available(propose_board, _params(SHEEP, WOOD, partner))[0]
        )
    # Player 1's hand is empty: no chance the trade could complete.
    assert not bool(propose_trade_available(propose_board, _params(SHEEP, WOOD, 1))[0])


def test_invalid_resource_choices(propose_board: Board) -> None:
    # Like-for-like, a give card the proposer lacks, and a packed index out of range.
    assert not bool(propose_trade_available(propose_board, _params(SHEEP, SHEEP, 2))[0])
    assert not bool(propose_trade_available(propose_board, _params(WHEAT, WOOD, 2))[0])
    assert not bool(
        propose_trade_available(propose_board, (jnp.array([25]), jnp.array([2])))[0]
    )


def test_invalid_wrong_phase(propose_board: Board) -> None:
    layout, st = propose_board
    st = st._replace(has_rolled=st.has_rolled.at[0].set(0))
    _, result = propose_trade_step((layout, st), _params(SHEEP, WOOD, 2))
    assert int(result[0]) == ActionResult.INVALID.value
