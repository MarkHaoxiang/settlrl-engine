"""Tests for the vectorized MaritimeTrade action."""

import jax.numpy as jnp
import numpy as np
from expecttest import assert_expected_inline

from catan_engine.mechanics.action import ActionResult, MaritimeTrade
from catan_engine.board import Board, give, make_board, place_settlement, to_main
from catan_engine.board.layout import PORT_V
from catan_engine.board.port import Port
from catan_engine.board.resources import BANK_INITIAL
from tests.mechanics.actions.fixtures import fmt

_PORT_V = np.asarray(PORT_V)


def _port_board(slot: int, port_type: int) -> Board:
    """MAIN board where player 0 owns a settlement on port ``slot`` set to ``port_type``."""
    layout, st = to_main(make_board())
    layout = layout._replace(
        port_allocation=layout.port_allocation.at[0, slot].set(int(port_type))
    )
    return place_settlement((layout, st), 0, int(_PORT_V[slot, 0]))


def test_success(trade_board: Board) -> None:
    state, result = MaritimeTrade()(trade_board, (jnp.array([0]), jnp.array([1])))
    assert_expected_inline(
        fmt(
            result,
            sheep=int(state.player_resources[0, 0, 0]),
            wheat=int(state.player_resources[0, 0, 1]),
        ),
        """\
result=OK
sheep=0
wheat=1""",
    )


def test_success_specific_port_2to1() -> None:
    # Player 0 owns a SHEEP (2:1) port and 2 sheep -> one wheat for two sheep.
    board = _port_board(slot=0, port_type=Port.SHEEP)
    board = give(board, 0, [2, 0, 0, 0, 0])
    state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
    assert_expected_inline(
        fmt(
            result,
            sheep=int(state.player_resources[0, 0, 0]),
            wheat=int(state.player_resources[0, 0, 1]),
        ),
        """\
result=OK
sheep=0
wheat=1""",
    )


def test_success_general_port_3to1() -> None:
    # Player 0 owns a 3:1 general port and 3 sheep -> one wheat for three sheep.
    board = _port_board(slot=1, port_type=Port.GENERAL)
    board = give(board, 0, [3, 0, 0, 0, 0])
    state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
    assert_expected_inline(
        fmt(
            result,
            sheep=int(state.player_resources[0, 0, 0]),
            wheat=int(state.player_resources[0, 0, 1]),
        ),
        """\
result=OK
sheep=0
wheat=1""",
    )


# Wrong-phase rejection is covered by the parametrized test in
# test_invalid_paths.py.


def test_invalid_bank_empty_for_received() -> None:
    # The bank holds zero wheat (all 19 sit in player hands), so trading *for*
    # wheat is rejected by the bank_stock >= 1 gate even with the give covered.
    board = to_main(make_board())
    board = give(board, 0, [4, int(BANK_INITIAL), 0, 0, 0])  # 4 sheep + all wheat
    before = np.asarray(board[1].player_resources)
    state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_give_equals_receive(trade_board: Board) -> None:
    before = np.asarray(trade_board[1].player_resources)
    state, result = MaritimeTrade()(trade_board, (jnp.array([0]), jnp.array([0])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_insufficient_resources() -> None:
    board = to_main(make_board())
    board = give(board, 0, [3, 0, 0, 0, 0])  # only 3 sheep, ratio is 4
    before = np.asarray(board[1].player_resources)
    state, result = MaritimeTrade()(board, (jnp.array([0]), jnp.array([1])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)
