"""Tests for state.py: the saturating cast, branchless select, and the fresh
game initialiser."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from catan_engine.board.dev_cards import DEV_CARD_COUNTS
from catan_engine.board.layout import N_VERTICES
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import (
    NO_INDEX,
    GamePhase,
    make_board_state,
    to_u8,
    tree_select,
)


class TestToU8:
    def test_saturates_both_ends(self) -> None:
        out = to_u8(jnp.array([-5, 0, 100, 255, 300, 1000], dtype=jnp.int32))
        assert out.dtype == jnp.uint8
        assert np.array_equal(np.asarray(out), [0, 0, 100, 255, 255, 255])

    def test_does_not_wrap(self) -> None:
        # 256 would wrap to 0 under a plain uint8 cast; saturation pins it at 255.
        assert int(to_u8(jnp.int32(256))) == 255


class TestTreeSelect:
    def test_true_picks_a_false_picks_b(self) -> None:
        a = make_board_state(1, key=jax.random.key(1))
        b = make_board_state(1, key=jax.random.key(2))
        a = a._replace(current_player=jnp.array([3], jnp.uint8))
        b = b._replace(current_player=jnp.array([1], jnp.uint8))

        picked_a = tree_select(jnp.bool_(True), a, b)
        picked_b = tree_select(jnp.bool_(False), a, b)
        assert int(picked_a.current_player[0]) == 3
        assert int(picked_b.current_player[0]) == 1

    def test_selects_every_leaf(self) -> None:
        a = make_board_state(1, key=jax.random.key(1))
        b = a._replace(
            vertex_owner=a.vertex_owner.at[0, 0].set(2),
            edge_road=a.edge_road.at[0, 0].set(2),
        )
        out = tree_select(jnp.bool_(False), a, b)
        assert int(out.vertex_owner[0, 0]) == 2
        assert int(out.edge_road[0, 0]) == 2


class TestMakeBoardState:
    def test_starts_in_setup(self) -> None:
        s = make_board_state(1)
        assert int(s.phase[0]) == GamePhase.SETUP_SETTLEMENT
        assert int(s.current_player[0]) == 0
        assert int(s.has_rolled[0]) == 0

    def test_no_holdings_or_points(self) -> None:
        s = make_board_state(1)
        assert s.player_resources.shape == (1, N_PLAYERS, N_RESOURCES)
        assert int(s.player_resources.sum()) == 0
        assert int(s.victory_points.sum()) == 0

    def test_full_dev_deck(self) -> None:
        s = make_board_state(1)
        assert np.array_equal(np.asarray(s.dev_deck[0]), DEV_CARD_COUNTS)

    def test_awards_unclaimed(self) -> None:
        s = make_board_state(1)
        assert int(s.longest_road_owner[0]) == NO_INDEX
        assert int(s.largest_army_owner[0]) == NO_INDEX

    def test_batch_size_leads_every_array(self) -> None:
        s = make_board_state(4)
        assert s.vertex_owner.shape == (4, N_VERTICES)
        assert s.player_resources.shape == (4, N_PLAYERS, N_RESOURCES)
        assert s.key.shape == (4,)

    def test_n_players_defaults_full_and_validates(self) -> None:
        # The per-player arrays are sized to the seated player count, and the
        # n_players property reads it back off the player axis.
        assert make_board_state(1).n_players == N_PLAYERS
        s = make_board_state(2, n_players=2)
        assert s.n_players == 2
        assert s.player_resources.shape == (2, 2, N_RESOURCES)
        assert s.victory_points.shape == (2, 2)
        for bad in (1, N_PLAYERS + 1):
            with pytest.raises(ValueError, match="n_players"):
                make_board_state(1, n_players=bad)
