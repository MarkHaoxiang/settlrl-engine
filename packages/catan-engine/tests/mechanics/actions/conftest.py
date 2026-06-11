"""Pytest fixtures for the per-action tests.

Each ``*_board`` fixture builds the legal mid-game position that action's success
case needs (composing the ``board.py`` helpers); invalid-case tests take the same
fixture and mutate it. ``render`` returns a helper that turns a post-action
``(layout, state)`` into the shared ASCII board snapshot used by expect tests.
Boards are batch=1 (single game, fixed seed) so success cases read out of lane 0.
"""

from __future__ import annotations

from collections.abc import Callable

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
    set_robber,
    to_main,
)
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.layout import TILE_V, BoardLayout
from catan_engine.board.state import BoardState, GamePhase
from catan_engine.mechanics.trade import pack_trade_single, propose_trade_step

from tests.mechanics.actions.fixtures import road_fixture, settlement_fixture
from tests.render import BoardRenderer

_TILE_V = np.asarray(TILE_V)


@pytest.fixture
def render() -> Callable[..., str]:
    """Render a board to text: the ASCII map by default, full tables if ``full``."""

    def _render(layout: BoardLayout, state: BoardState, *, full: bool = False) -> str:
        renderer = BoardRenderer(layout, state)
        return str(renderer) if full else renderer.render_map()

    return _render


@pytest.fixture
def road_board() -> tuple[Board, int]:
    """MAIN board; player 0 owns a settlement + one road's worth -> (board, edge)."""
    return road_fixture()


@pytest.fixture
def settlement_board() -> tuple[Board, int]:
    """MAIN board with a 2-road spur off vertex 0 -> (board, legal vertex)."""
    return settlement_fixture()


@pytest.fixture
def city_board() -> tuple[Board, int]:
    """MAIN board; player 0's settlement at vertex 0 with one city's worth."""
    board = to_main(make_board())
    board = place_settlement(board, 0, 0)
    board = give(board, 0, [0, 2, 0, 0, 3])  # 2 wheat + 3 ore
    return board, 0


@pytest.fixture
def buy_board() -> Board:
    """MAIN board (fixed seed) where player 0 can afford exactly one dev card."""
    board = to_main(make_board(seed=0))
    return give(board, 0, [1, 1, 0, 0, 1])  # sheep + wheat + ore


@pytest.fixture
def roll_board() -> Board:
    """ROLL-phase board (fixed seed -> deterministic dice) with one settlement."""
    board = set_phase(make_board(seed=0), GamePhase.ROLL)
    return place_settlement(board, 0, 0)


@pytest.fixture
def discard_board() -> Callable[..., Board]:
    """Factory: DISCARD board where player 0 holds 8 cards and owes ``owed``."""

    def _make(owed: int = 4) -> Board:
        board = set_phase(make_board(seed=0), GamePhase.DISCARD)
        board = give(board, 0, [4, 4, 0, 0, 0])  # 8 cards
        layout, st = board
        st = st._replace(pending_discard=st.pending_discard.at[0, 0].set(owed))
        return (layout, st)

    return _make


@pytest.fixture
def trade_board() -> Board:
    """MAIN board where player 0 holds 4 sheep (no port -> 4:1 bank trade)."""
    board = to_main(make_board())
    return give(board, 0, [4, 0, 0, 0, 0])


@pytest.fixture
def propose_board() -> Board:
    """MAIN board (4p): player 0 holds 1 sheep, player 2 holds 3 wood, player 1
    holds nothing -- the domestic-trade success and invalid cases."""
    board = to_main(make_board(seed=0))
    board = give(board, 0, [1, 0, 0, 0, 0])
    return give(board, 2, [0, 0, 3, 0, 0])


@pytest.fixture
def response_board(propose_board: Board) -> Board:
    """TRADE_RESPONSE board: player 0 has offered player 2 a sheep for a wood."""
    idx, target = pack_trade_single(0, 2, partner=2)
    state, _ = propose_trade_step(
        propose_board, (jnp.array([idx]), jnp.array([target]))
    )
    return (propose_board[0], state)


@pytest.fixture
def robber_board() -> Board:
    """MOVE_ROBBER board: player 1 sits on tile 0 with 1 sheep; robber elsewhere."""
    board = to_main(make_board(seed=0))
    board = set_phase(board, GamePhase.MOVE_ROBBER)
    board = place_settlement(board, 1, int(_TILE_V[0, 0]))
    board = give(board, 1, [1, 0, 0, 0, 0])  # 1 sheep to steal
    board = set_robber(board, 1 % _TILE_V.shape[0])
    return board


@pytest.fixture
def knight_board() -> Board:
    """MAIN board: player 0 holds a Knight; player 1 sits on tile 0 with 1 sheep."""
    board = to_main(make_board(seed=0))
    board = give_dev_card(board, 0, DevCard.KNIGHT)
    board = place_settlement(board, 1, int(_TILE_V[0, 0]))
    board = give(board, 1, [1, 0, 0, 0, 0])  # 1 sheep to steal
    board = set_robber(board, 1 % _TILE_V.shape[0])
    return board


@pytest.fixture
def monopoly_board() -> Board:
    """MAIN board: player 0 holds Monopoly; players 0/1/2 hold 1/3/2 sheep."""
    board = to_main(make_board())
    board = give(board, 0, [1, 0, 0, 0, 0])
    board = give(board, 1, [3, 0, 0, 0, 0])
    board = give(board, 2, [2, 0, 0, 0, 0])
    return give_dev_card(board, 0, DevCard.MONOPOLY)


@pytest.fixture
def road_building_board() -> Board:
    """MAIN board where player 0 holds a Road Building card."""
    board = to_main(make_board())
    return give_dev_card(board, 0, DevCard.ROAD_BUILDING)


@pytest.fixture
def yop_board() -> Board:
    """MAIN board where player 0 holds a Year of Plenty card and no resources."""
    board = to_main(make_board())
    return give_dev_card(board, 0, DevCard.YEAR_OF_PLENTY)


@pytest.fixture
def setup_board() -> Board:
    """Fresh SETUP_SETTLEMENT board (fixed seed)."""
    return make_board(seed=0)


@pytest.fixture
def setup_road_board() -> Board:
    """SETUP_ROAD board: player 0 just placed a settlement at vertex 0."""
    board = set_phase(make_board(seed=0), GamePhase.SETUP_ROAD)
    return place_settlement(board, 0, 0)
