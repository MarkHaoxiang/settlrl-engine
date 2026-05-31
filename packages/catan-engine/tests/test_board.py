"""Tests for board.py: the action-agnostic construction / shaping helpers that
the rest of the test suite composes its fixtures from."""

from __future__ import annotations

import numpy as np

from catan_engine.board import (
    give,
    give_dev_card,
    make_board,
    place_city,
    place_road,
    place_settlement,
    replicate,
    set_phase,
    set_robber,
    to_main,
)
from catan_engine.dev_cards import DevCard
from catan_engine.layout import N_TILES, N_VERTICES
from catan_engine.resources import N_PLAYERS, N_RESOURCES
from catan_engine.state import GamePhase


class TestMakeBoardAndReplicate:
    def test_make_board_shapes(self) -> None:
        layout, state = make_board(1, seed=0)
        assert state.vertex_owner.shape == (1, N_VERTICES)
        assert layout.tile_resource.shape == (1, N_TILES)

    def test_seed_is_deterministic(self) -> None:
        a, _ = make_board(1, seed=7)
        b, _ = make_board(1, seed=7)
        assert np.array_equal(
            np.asarray(a.tile_resource), np.asarray(b.tile_resource)
        )

    def test_replicate_broadcasts_and_copies(self) -> None:
        board = make_board(1, seed=0)
        layout, state = replicate(board, 5)
        assert state.vertex_owner.shape == (5, N_VERTICES)
        assert layout.tile_resource.shape == (5, N_TILES)
        # Every lane is a copy of the single source game.
        assert np.array_equal(
            np.asarray(layout.tile_resource[0]), np.asarray(layout.tile_resource[4])
        )


class TestPhaseHelpers:
    def test_set_phase(self) -> None:
        _, state = set_phase(make_board(seed=0), GamePhase.ROLL)
        assert int(state.phase[0]) == GamePhase.ROLL

    def test_to_main_marks_rolled_player(self) -> None:
        _, state = to_main(make_board(seed=0), player=2)
        assert int(state.phase[0]) == GamePhase.MAIN
        assert int(state.has_rolled[0]) == 1
        assert int(state.current_player[0]) == 2


class TestOccupancyHelpers:
    def test_give_sets_hand(self) -> None:
        _, state = give(make_board(seed=0), 0, [1, 2, 3, 4, 0])
        assert np.array_equal(
            np.asarray(state.player_resources[0, 0]), [1, 2, 3, 4, 0]
        )
        assert state.player_resources.shape == (1, N_PLAYERS, N_RESOURCES)

    def test_place_settlement(self) -> None:
        _, state = place_settlement(make_board(seed=0), 0, 5)
        assert int(state.vertex_owner[0, 5]) == 1  # player + 1
        assert int(state.vertex_type[0, 5]) == 1
        assert int(state.victory_points[0, 0]) == 1

    def test_place_road(self) -> None:
        _, state = place_road(make_board(seed=0), 1, 9)
        assert int(state.edge_road[0, 9]) == 2  # player + 1

    def test_place_city_fresh_is_two_points(self) -> None:
        _, state = place_city(make_board(seed=0), 0, 5)
        assert int(state.vertex_type[0, 5]) == 2
        assert int(state.victory_points[0, 0]) == 2

    def test_place_city_upgrade_adds_one_point(self) -> None:
        board = place_settlement(make_board(seed=0), 0, 5)  # +1 VP
        _, state = place_city(board, 0, 5)  # upgrade: +1 more, not +2
        assert int(state.vertex_type[0, 5]) == 2
        assert int(state.victory_points[0, 0]) == 2

    def test_give_dev_card(self) -> None:
        _, state = give_dev_card(make_board(seed=0), 0, DevCard.KNIGHT, count=2)
        assert int(state.dev_hand[0, 0, DevCard.KNIGHT]) == 2

    def test_set_robber(self) -> None:
        _, state = set_robber(make_board(seed=0), 4)
        assert int(state.robber[0]) == 4
