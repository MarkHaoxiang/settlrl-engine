"""Parametrized invalid-path tests shared across the build / play / trade actions.

These consolidate the near-identical per-file ``test_invalid_wrong_phase`` and
``test_invalid_cannot_afford`` / ``test_invalid_no_card`` cases: the same shape
(set up a legal board, break one precondition, assert INVALID and that the
relevant state array is left untouched) repeated for every action. Each action
keeps its own representative *success* expect-test in its own file; only the
copy-paste invalidation cases live here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    place_settlement,
    set_phase,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.state import BoardState, GamePhase
from catan_engine.mechanics.action import ActionResult
from catan_engine.mechanics.common import ResultCode
from catan_engine.mechanics.development import (
    buy_development_card_step,
    play_monopoly_step,
    play_road_building_step,
    play_year_of_plenty_step,
)
from catan_engine.mechanics.placement import (
    build_city_step,
    build_road_step,
    build_settlement_step,
)
from catan_engine.mechanics.trade import maritime_step

from tests.mechanics.actions.fixtures import road_fixture, settlement_fixture

# A batched action step: (board, params) -> (new state, ActionResult codes).
StepFn = Callable[[Board, Any], tuple[BoardState, ResultCode]]


def _road() -> tuple[Board, object]:
    board, edge = road_fixture()
    return board, jnp.array([edge])


def _settlement() -> tuple[Board, object]:
    board, vertex = settlement_fixture()
    return board, jnp.array([vertex])


def _city() -> tuple[Board, object]:
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    board = give(board, 0, [0, 2, 0, 0, 3])  # 2 wheat + 3 ore
    return board, jnp.array([0])


def _buy() -> tuple[Board, object]:
    board = to_main(make_board(seed=0))
    board = give(board, 0, [1, 1, 0, 0, 1])  # sheep + wheat + ore
    return board, None


def _maritime() -> tuple[Board, object]:
    board = to_main(make_board())
    board = give(board, 0, [4, 0, 0, 0, 0])  # 4 sheep, no port -> 4:1
    return board, (jnp.array([0]), jnp.array([1]))


def _road_building() -> tuple[Board, object]:
    board = to_main(make_board())
    board = give_dev_card(board, 0, DevCard.ROAD_BUILDING)
    return board, None


def _monopoly() -> tuple[Board, object]:
    board = to_main(make_board())
    board = give(board, 1, [3, 0, 0, 0, 0])
    board = give_dev_card(board, 0, DevCard.MONOPOLY)
    return board, jnp.array([0])


def _year_of_plenty() -> tuple[Board, object]:
    board = to_main(make_board())
    board = give_dev_card(board, 0, DevCard.YEAR_OF_PLENTY)
    return board, (jnp.array([2]), jnp.array([3]))


# (id, action, board+params builder, inspected state field whose array must be
# left untouched when the action is rejected).
_WRONG_PHASE_CASES = [
    ("build_road", build_road_step, _road, "edge_road"),
    ("build_settlement", build_settlement_step, _settlement, "vertex_owner"),
    ("build_city", build_city_step, _city, "vertex_type"),
    ("buy_development_card", buy_development_card_step, _buy, "dev_deck"),
    ("maritime_trade", maritime_step, _maritime, "player_resources"),
    ("play_road_building", play_road_building_step, _road_building, "dev_hand"),
    ("play_monopoly", play_monopoly_step, _monopoly, "player_resources"),
    (
        "play_year_of_plenty",
        play_year_of_plenty_step,
        _year_of_plenty,
        "player_resources",
    ),
]


@pytest.mark.parametrize(
    ("action", "build", "field"),
    [(a, b, f) for _, a, b, f in _WRONG_PHASE_CASES],
    ids=[c[0] for c in _WRONG_PHASE_CASES],
)
def test_invalid_wrong_phase(
    action: StepFn,
    build: Callable[[], tuple[Board, object]],
    field: str,
) -> None:
    """A MAIN-phase-only action attempted in ROLL is INVALID and a no-op."""
    board, params = build()
    board = set_phase(board, GamePhase.ROLL)
    before = np.asarray(getattr(board[1], field))
    state, result = action(board, params)
    assert int(result[0]) == ActionResult.INVALID.value
    assert np.array_equal(np.asarray(getattr(state, field)), before)


# The build / buy actions share the same "strip the hand -> can't pay" shape.
_CANNOT_AFFORD_CASES = [
    ("build_road", build_road_step, _road),
    ("build_settlement", build_settlement_step, _settlement),
    ("build_city", build_city_step, _city),
    ("buy_development_card", buy_development_card_step, _buy),
]


@pytest.mark.parametrize(
    ("action", "build"),
    [(a, b) for _, a, b in _CANNOT_AFFORD_CASES],
    ids=[c[0] for c in _CANNOT_AFFORD_CASES],
)
def test_invalid_cannot_afford(
    action: StepFn,
    build: Callable[[], tuple[Board, object]],
) -> None:
    """With an empty hand (and no free roads) the cost gate rejects the build."""
    board, params = build()
    board = give(board, 0, [0, 0, 0, 0, 0])
    _, result = action(board, params)
    assert int(result[0]) == ActionResult.INVALID.value
