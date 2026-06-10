"""Tests for development.py: dev-card playability and the weighted draw."""

from __future__ import annotations

from typing import TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from catan_engine.board.dev_cards import DevCard
from catan_engine.board.state import make_board_state
from catan_engine.mechanics import development

_T = TypeVar("_T")


def _single(tree: _T) -> _T:
    return cast(_T, jax.tree_util.tree_map(lambda x: x[0], tree))


class TestPlayableDev:
    def test_held_and_not_bought_is_playable(self) -> None:
        state = make_board_state(1)
        state = state._replace(dev_hand=state.dev_hand.at[0, 0, DevCard.KNIGHT].set(1))
        assert bool(
            development.playable_dev(_single(state), jnp.int32(0), DevCard.KNIGHT)
        )

    def test_bought_this_turn_is_not_playable(self) -> None:
        state = make_board_state(1)
        state = state._replace(
            dev_hand=state.dev_hand.at[0, 0, DevCard.KNIGHT].set(1),
            dev_bought=state.dev_bought.at[0, DevCard.KNIGHT].set(1),
        )
        assert not bool(
            development.playable_dev(_single(state), jnp.int32(0), DevCard.KNIGHT)
        )

    def test_none_held_is_not_playable(self) -> None:
        assert not bool(
            development.playable_dev(
                _single(make_board_state(1)), jnp.int32(0), DevCard.KNIGHT
            )
        )


class TestDrawDevCard:
    def test_only_draws_available_types(self) -> None:
        deck = jnp.array([0, 0, 5, 0, 0], jnp.uint8)  # only YEAR_OF_PLENTY remains
        for s in range(20):
            _, card = development.draw_dev_card(jax.random.key(s), deck)
            assert int(card) == DevCard.YEAR_OF_PLENTY

    def test_advances_key(self) -> None:
        deck = jnp.array([1, 1, 1, 1, 1], jnp.uint8)
        k_in = jax.random.key(0)
        k_out, _ = development.draw_dev_card(k_in, deck)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(k_out)),
            np.asarray(jax.random.key_data(k_in)),
        )
