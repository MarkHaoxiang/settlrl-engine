"""Consistency invariants of world sampling (``sample_world``).

A sample fills a censored state's hidden fields without disturbing anything
public: hand sizes, dev counts, per-type totals, and the observer's own rows
must all match the public record.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from catan_engine.belief import PlayerBelief
from catan_engine.board.dev_cards import DEV_CARD_COUNTS
from catan_engine.board.state import BoardState
from catan_engine.env import BatchedCatanEnv

from catan_agents import sample_world

_DEV_COUNTS = jnp.asarray(DEV_CARD_COUNTS, jnp.int32)


def _played_view(
    n_players: int, n_steps: int, seed: int = 0
) -> tuple[BoardState, PlayerBelief]:
    """A mid-game censored view: lane 0, observer 0, after random play."""
    env = BatchedCatanEnv(
        batch_size=4, seed=seed, n_players=n_players, track_beliefs=True
    )
    key = jax.random.key(seed)
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
    censored, pb = env.belief_view(0)
    return cast(
        "tuple[BoardState, PlayerBelief]",
        jax.tree.map(lambda x: x[0], (censored, pb)),
    )


@pytest.mark.parametrize("n_players", [2, 4])
def test_sample_matches_the_public_record(n_players: int) -> None:
    censored, pb = _played_view(n_players, n_steps=300)
    me = jnp.int32(0)
    for seed in range(10):
        world = sample_world(jax.random.key(seed), censored, pb, me)
        res = world.player_resources.astype(jnp.int32)
        # Public resource counts: sizes per player, totals per type.
        assert bool(jnp.all(res.sum(axis=1) == pb.hand_size))
        assert bool(jnp.all(res.sum(axis=0) == pb.res_total))
        # The proven floor is respected; own row is untouched.
        assert bool(jnp.all(pb.res_lo.astype(jnp.int32) <= res))
        assert bool(jnp.all(res[0] == censored.player_resources[0]))
        # Dev cards: public counts and deck conservation.
        dev = world.dev_hand.astype(jnp.int32)
        assert bool(jnp.all(dev.sum(axis=1) == pb.dev_count))
        assert bool(jnp.all(dev[0] == censored.dev_hand[0]))
        assert bool(
            jnp.all(
                world.dev_deck.astype(jnp.int32) + dev.sum(axis=0)
                == censored.dev_deck.astype(jnp.int32)
                + censored.dev_hand.astype(jnp.int32).sum(axis=0)
            )
        )
        # The constant censored key was replaced with a fresh one.
        assert not bool(
            jnp.all(
                jax.random.key_data(world.key)
                == jax.random.key_data(jax.random.key(0))
            )
        )


def test_sample_varies_with_the_key() -> None:
    # Grant the opponent three (hidden) dev cards: the deal must have freedom.
    censored, pb = _played_view(2, n_steps=0)
    pb = pb._replace(dev_count=pb.dev_count.at[1].set(3))
    me = jnp.int32(0)
    hands = {
        tuple(
            int(c)
            for c in sample_world(jax.random.key(s), censored, pb, me).dev_hand[1]
        )
        for s in range(30)
    }
    assert len(hands) > 1  # the posterior is actually being sampled


def test_two_player_resources_are_pinned() -> None:
    # 2p beliefs are exact on resources, so every sample agrees on them.
    censored, pb = _played_view(2, n_steps=300)
    me = jnp.int32(0)
    worlds = [
        sample_world(jax.random.key(s), censored, pb, me).player_resources
        for s in range(5)
    ]
    for w in worlds[1:]:
        assert bool(jnp.all(w == worlds[0]))
