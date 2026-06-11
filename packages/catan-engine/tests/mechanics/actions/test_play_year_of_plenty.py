"""Tests for the vectorized PlayYearOfPlenty action."""

import jax.numpy as jnp
import numpy as np
from catan_engine.board import Board, give_dev_card, make_board, set_phase, to_main
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.state import GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.development import play_year_of_plenty_step
from expecttest import assert_expected_inline

from tests.mechanics.actions.fixtures import fmt


def test_success(yop_board: Board) -> None:
    # Take wood (2) and brick (3).
    state, result = play_year_of_plenty_step(
        yop_board, (jnp.array([2]), jnp.array([3]))
    )
    assert_expected_inline(
        fmt(
            result,
            wood=int(state.player_resources[0, 0, 2]),
            brick=int(state.player_resources[0, 0, 3]),
            dev_played=int(state.dev_played[0]),
            yop_hand=int(state.dev_hand[0, 0, DevCard.YEAR_OF_PLENTY]),
        ),
        """\
result=OK
wood=1
brick=1
dev_played=1
yop_hand=0""",
    )


def test_success_same_resource(yop_board: Board) -> None:
    # a == b: take two sheep (0).
    state, result = play_year_of_plenty_step(
        yop_board, (jnp.array([0]), jnp.array([0]))
    )
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.player_resources[0, 0, 0]) == 2


# Wrong-phase rejection is covered by the parametrized test in
# test_invalid_paths.py.


def test_invalid_no_card() -> None:
    board = to_main(make_board())  # no Year of Plenty card
    before = np.asarray(board[1].player_resources)
    state, result = play_year_of_plenty_step(board, (jnp.array([2]), jnp.array([3])))
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_dev_already_played(yop_board: Board) -> None:
    layout, st = yop_board
    st = st._replace(dev_played=st.dev_played.at[0].set(1))
    before = np.asarray(st.player_resources)
    state, result = play_year_of_plenty_step(
        (layout, st), (jnp.array([2]), jnp.array([3]))
    )
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_invalid_out_of_range(yop_board: Board) -> None:
    before = np.asarray(yop_board[1].player_resources)
    state, result = play_year_of_plenty_step(
        yop_board, (jnp.array([7]), jnp.array([3]))
    )
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(state.player_resources), before)


def test_playable_before_the_roll() -> None:
    # Rulebook: the one dev card may be played any time during the turn.
    board = set_phase(make_board(seed=0), GamePhase.ROLL)
    board = give_dev_card(board, 0, DevCard.YEAR_OF_PLENTY)
    state, result = play_year_of_plenty_step(board, (jnp.array([2]), jnp.array([3])))
    assert int(result[0]) == ActionResult.SUCCESS.value
    assert int(state.phase[0]) == GamePhase.ROLL  # the roll is still owed
