"""Tests for the vectorized ProposeTrade action (bundle domestic-trade offer)."""

import jax
import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline
from settlrl_engine.board import Board, give, make_board, to_main
from settlrl_engine.board.state import GamePhase
from settlrl_engine.mechanics.action import ActionResult
from settlrl_engine.mechanics.common import agent_selection_single
from settlrl_engine.mechanics.trade import (
    pack_trade,
    pack_trade_single,
    propose_trade_available,
    propose_trade_step,
)

from tests.mechanics.actions.fixtures import fmt

SHEEP, WHEAT, WOOD = 0, 1, 2
Params = tuple[jnp.ndarray, jnp.ndarray]


def _params(give_r: int, receive_r: int, partner: int) -> Params:
    idx, target = pack_trade_single(give_r, receive_r, partner)
    return jnp.array([idx]), jnp.array([target])


def _bundle(give: list[int], receive: list[int], partner: int) -> Params:
    idx, target = pack_trade(give, receive, partner)
    return jnp.array([idx]), jnp.array([target])


def test_propose_parks_game_in_trade_response(propose_board: Board) -> None:
    state, result = propose_trade_step(propose_board, _params(SHEEP, WOOD, 2))
    assert_expected_inline(
        fmt(
            result,
            phase=str(GamePhase(int(state.phase[0]))),
            partner=int(state.trade_partner[0]),
            give=np.asarray(state.trade_give[0]).tolist(),
            receive=np.asarray(state.trade_receive[0]).tolist(),
            acting=int(jax.vmap(agent_selection_single)(state)[0]),
        ),
        """\
result=OK
phase=TRADE_RESPONSE
partner=2
give=[1, 0, 0, 0, 0]
receive=[0, 0, 1, 0, 0]
acting=2""",
    )
    # The offer itself moves no cards.
    assert np.array_equal(
        np.asarray(state.player_resources),
        np.asarray(propose_board[1].player_resources),
    )


def test_bundle_offer_records_the_counts(propose_board: Board) -> None:
    # 1 sheep for 2 wood + 1 brick: any multiset goes through the packed params.
    board = give(propose_board, 2, [0, 0, 2, 1, 0])
    state, result = propose_trade_step(
        board, _bundle([1, 0, 0, 0, 0], [0, 0, 2, 1, 0], partner=2)
    )
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert np.asarray(state.trade_give[0]).tolist() == [1, 0, 0, 0, 0]
    assert np.asarray(state.trade_receive[0]).tolist() == [0, 0, 2, 1, 0]


def test_propose_ignores_partner_exact_holdings(propose_board: Board) -> None:
    # Player 2 holds no ore, but proposing sheep -> ore is still legal: only
    # public information (a big-enough hand) gates the offer.
    assert bool(propose_trade_available(propose_board, _params(SHEEP, 4, 2))[0])
    # Asking for more cards than the partner holds *at all* is not.
    assert not bool(
        propose_trade_available(
            propose_board, _bundle([1, 0, 0, 0, 0], [0, 0, 0, 0, 4], partner=2)
        )[0]
    )


def test_invalid_two_player_board() -> None:
    board = to_main(make_board(seed=0, n_players=2))
    board = give(board, 0, [1, 0, 0, 0, 0])
    board = give(board, 1, [0, 0, 3, 0, 0])
    state, result = propose_trade_step(board, _params(SHEEP, WOOD, 1))
    assert int(result[0]) == ActionResult.INVALID.value
    assert int(state.phase[0]) == GamePhase.MAIN


def test_invalid_partner_choices(propose_board: Board) -> None:
    # Self, and a partner with an empty hand (no chance the trade completes).
    assert not bool(propose_trade_available(propose_board, _params(SHEEP, WOOD, 0))[0])
    assert not bool(propose_trade_available(propose_board, _params(SHEEP, WOOD, 1))[0])
    # Out-of-range packed params.
    assert not bool(
        propose_trade_available(propose_board, (jnp.array([1]), jnp.array([-1])))[0]
    )


def test_invalid_bundle_shapes(propose_board: Board) -> None:
    # Like-for-like overlap, a give bundle the proposer lacks, and gifts
    # (one side empty) are all rejected.
    assert not bool(propose_trade_available(propose_board, _params(SHEEP, SHEEP, 2))[0])
    assert not bool(propose_trade_available(propose_board, _params(WHEAT, WOOD, 2))[0])
    assert not bool(
        propose_trade_available(
            propose_board, _bundle([2, 0, 0, 0, 0], [0, 0, 1, 0, 0], partner=2)
        )[0]
    )
    assert not bool(
        propose_trade_available(
            propose_board, _bundle([1, 0, 0, 0, 0], [0, 0, 0, 0, 0], partner=2)
        )[0]
    )
    assert not bool(
        propose_trade_available(
            propose_board, _bundle([0, 0, 0, 0, 0], [0, 0, 1, 0, 0], partner=2)
        )[0]
    )
    assert not bool(
        propose_trade_available(
            propose_board, _bundle([1, 0, 1, 0, 0], [0, 0, 1, 0, 0], partner=2)
        )[0]
    )


def test_invalid_wrong_phase(propose_board: Board) -> None:
    layout, st = propose_board
    st = st._replace(has_rolled=st.has_rolled.at[0].set(0))
    _, result = propose_trade_step((layout, st), _params(SHEEP, WOOD, 2))
    assert int(result[0]) == ActionResult.INVALID.value
