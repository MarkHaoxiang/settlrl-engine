"""Tests for the vectorized RollDice action."""

import jax
import numpy as np
from expecttest import assert_expected_inline
from settlrl_engine.board import Board, give, make_board, to_main
from settlrl_engine.board.state import BoardState, GamePhase
from settlrl_engine.mechanics.action import ActionResult
from settlrl_engine.mechanics.dice import roll_step

from tests.mechanics.actions.fixtures import fmt


def test_success(roll_board: Board) -> None:
    state, result = roll_step(roll_board, None)
    assert_expected_inline(
        fmt(
            result,
            dice=int(state.dice_roll[0]),
            phase=str(GamePhase(int(state.phase[0]))),
            has_rolled=int(state.has_rolled[0]),
        ),
        """\
result=OK
dice=4
phase=MAIN
has_rolled=1""",
    )


def test_invalid_not_roll_phase() -> None:
    board = to_main(make_board(seed=0))  # MAIN phase -> cannot roll
    before = np.asarray(board[1].phase)
    state, result = roll_step(board, None)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.phase), before)


def test_seven_routes_to_discard_and_suppresses_production(roll_board: Board) -> None:
    # PRNG key 0 yields a deterministic roll of 7 (see test_dice.roll_dice).
    # Player 0 holds 10 cards, so on a 7 they owe 5 and the phase must route to
    # DISCARD; production for the (irrelevant) tiles is suppressed.
    layout, st = roll_board
    st = st._replace(key=jax.random.key(0)[None])
    board: Board = (layout, st)
    board = give(board, 0, [4, 4, 1, 1, 0])  # 10 cards -> owes 5
    before = np.asarray(board[1].player_resources)

    state, result = roll_step(board, None)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.dice_roll[0]) == 7
    assert int(state.phase[0]) == GamePhase.DISCARD
    assert int(state.pending_discard[0, 0]) == 5  # 10 // 2
    assert int(state.pending_discard[0].sum()) == 5  # only player 0 owes
    # 7 suppresses all production: hands are unchanged.
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_already_rolled(roll_board: Board) -> None:
    layout, st = roll_board
    st = st._replace(has_rolled=st.has_rolled.at[0].set(1))
    _, result = roll_step((layout, st), None)
    assert int(result[0]) == ActionResult.INVALID.value


def test_forced_outcome(roll_board: Board) -> None:
    """params=r (2..12) forces the roll; the payout matches the sampled path
    for the same number and the key advances identically."""
    sampled, _ = roll_step(roll_board, None)  # seed's natural roll is 4
    forced, result = roll_step(roll_board, 4)
    assert int(result[0]) == ActionResult.SUCCESS.value

    def raw(st: BoardState) -> BoardState:
        return st._replace(key=jax.random.key_data(st.key))

    for a, b in zip(raw(forced), raw(sampled), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_forced_seven_routes_to_robber(roll_board: Board) -> None:
    state, result = roll_step(roll_board, 7)
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.dice_roll[0]) == 7
    assert int(state.phase[0]) == GamePhase.MOVE_ROBBER  # no hand over 7 cards
