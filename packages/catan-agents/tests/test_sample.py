"""Consistency invariants of world sampling (``sample_world``).

A sample rebuilds a playable ``BoardState`` from a ``BeliefView`` without
disturbing anything public: the public fields are copied through, and hand
sizes, dev counts, per-type totals, and the observer's own rows must all match
the public record.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from catan_engine.belief import BeliefView, PublicState
from catan_engine.env import BatchedCatanEnv

from catan_agents import sample_world


def _played_view(n_players: int, n_steps: int, seed: int = 0) -> BeliefView:
    """A mid-game view: lane 0, observer 0, after random play."""
    env = BatchedCatanEnv(
        batch_size=4, seed=seed, n_players=n_players, track_beliefs=True
    )
    key = jax.random.key(seed)
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    return cast(BeliefView, jax.tree.map(lambda x: x[0], env.belief_view(0)))


@pytest.mark.parametrize("n_players", [2, 4])
def test_sample_matches_the_public_record(n_players: int) -> None:
    view = _played_view(n_players, n_steps=300)
    pb = view.belief
    me = jnp.int32(0)
    keys = set()
    for seed in range(10):
        world = sample_world(jax.random.key(seed), view, me)
        # The public fields are copied through untouched.
        for name in PublicState._fields:
            assert bool(jnp.all(getattr(world, name) == getattr(view.public, name)))
        res = world.player_resources.astype(jnp.int32)
        # Public resource counts: sizes per player, totals per type.
        assert bool(jnp.all(res.sum(axis=1) == pb.hand_size))
        assert bool(jnp.all(res.sum(axis=0) == pb.res_total))
        # The proven floor is respected; own row is exact (lo is pinned there).
        assert bool(jnp.all(pb.res_lo.astype(jnp.int32) <= res))
        assert bool(jnp.all(res[0] == pb.res_lo[0]))
        # Dev cards: public counts, own hand, and pool conservation.
        dev = world.dev_hand.astype(jnp.int32)
        assert bool(jnp.all(dev.sum(axis=1) == pb.dev_count))
        assert bool(jnp.all(dev[0] == view.own_dev))
        assert bool(
            jnp.all(
                world.dev_deck.astype(jnp.int32) + dev.sum(axis=0)
                == view.unseen_dev.astype(jnp.int32) + view.own_dev.astype(jnp.int32)
            )
        )
        keys.add(tuple(int(x) for x in jax.random.key_data(world.key).ravel()))
    # Each sample carries its own fresh PRNG key.
    assert len(keys) == 10


def test_sample_varies_with_the_key() -> None:
    # Grant the opponent three (hidden) dev cards: the deal must have freedom.
    view = _played_view(2, n_steps=0)
    view = view._replace(
        belief=view.belief._replace(dev_count=view.belief.dev_count.at[1].set(3))
    )
    me = jnp.int32(0)
    hands = {
        tuple(
            int(c) for c in sample_world(jax.random.key(s), view, me).dev_hand[1]
        )
        for s in range(30)
    }
    assert len(hands) > 1  # the posterior is actually being sampled


def test_two_player_resources_are_pinned() -> None:
    # 2p beliefs are exact on resources, so every sample agrees on them.
    view = _played_view(2, n_steps=300)
    me = jnp.int32(0)
    worlds = [
        sample_world(jax.random.key(s), view, me).player_resources for s in range(5)
    ]
    for w in worlds[1:]:
        assert bool(jnp.all(w == worlds[0]))
