"""Tests for dice.py: the two-die roll and a sanity check on production payout.

(The full distribute_resources / bank-cap equivalence sweep lives in
test_rules.py; here we cover roll_dice and the empty-board no-op.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine import dice
from catan_engine.board import make_board


class TestRollDice:
    def test_sum_in_range(self) -> None:
        for s in range(200):
            _, roll = dice.roll_dice(jax.random.key(s))
            assert 2 <= int(roll) <= 12

    def test_deterministic_for_a_key(self) -> None:
        key = jax.random.key(42)
        _, a = dice.roll_dice(key)
        _, b = dice.roll_dice(key)
        assert int(a) == int(b)

    def test_advances_key(self) -> None:
        k_in = jax.random.key(0)
        k_out, _ = dice.roll_dice(k_in)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(k_out)),
            np.asarray(jax.random.key_data(k_in)),
        )

    def test_covers_a_spread_of_totals(self) -> None:
        totals = {int(dice.roll_dice(jax.random.key(s))[1]) for s in range(200)}
        assert len(totals) >= 6  # not stuck on a single value


def test_distribute_on_empty_board_is_noop() -> None:
    layout, state = make_board(1, seed=0)
    single = jax.tree_util.tree_map(lambda x: x[0], state)
    layout1 = jax.tree_util.tree_map(lambda x: x[0], layout)
    for roll in range(2, 13):
        out = dice.distribute_resources(layout1, single, jnp.int32(roll))
        assert int(np.asarray(out.player_resources).sum()) == 0
