"""Conservation invariants of the root determinization (`redeal_dev_cards`)."""

import jax
import jax.numpy as jnp

from catan_engine.board.dev_cards import DEV_CARD_COUNTS
from catan_engine.env import BatchedCatanEnv

from catan_agents.two_player.belief import redeal_dev_cards


def test_redeal_preserves_everything_the_player_can_see() -> None:
    env = BatchedCatanEnv(batch_size=1, seed=0, n_players=2)
    _, state = jax.tree.map(lambda x: x[0], env.board)
    # Deal some cards out so the re-deal has work to do.
    hand = jnp.zeros_like(state.dev_hand)
    hand = hand.at[0].set(jnp.asarray([2, 0, 1, 0, 1], jnp.uint8))  # me
    hand = hand.at[1].set(jnp.asarray([1, 1, 0, 0, 2], jnp.uint8))  # opponent
    deck = jnp.asarray(DEV_CARD_COUNTS, jnp.uint8) - hand.sum(axis=0)
    state = state._replace(dev_hand=hand, dev_deck=deck)

    me = jnp.int32(0)
    pool_before = state.dev_deck + state.dev_hand[1]
    for seed in range(20):
        new = redeal_dev_cards(jax.random.key(seed), state, me)
        # Own hand untouched; opponent hand size unchanged; deck + opponent
        # hand is the same unseen pool; nothing goes negative.
        assert bool(jnp.all(new.dev_hand[0] == state.dev_hand[0]))
        assert int(new.dev_hand[1].sum()) == int(state.dev_hand[1].sum())
        assert bool(jnp.all(new.dev_deck + new.dev_hand[1] == pool_before))
        assert bool(jnp.all(new.dev_deck <= pool_before))


def test_redeal_varies_with_the_key() -> None:
    env = BatchedCatanEnv(batch_size=1, seed=0, n_players=2)
    _, state = jax.tree.map(lambda x: x[0], env.board)
    hand = jnp.zeros_like(state.dev_hand).at[1].set(
        jnp.asarray([2, 1, 0, 0, 0], jnp.uint8)
    )
    deck = jnp.asarray(DEV_CARD_COUNTS, jnp.uint8) - hand.sum(axis=0)
    state = state._replace(dev_hand=hand, dev_deck=deck)
    hands = {
        tuple(
            int(c)
            for c in redeal_dev_cards(jax.random.key(s), state, jnp.int32(0)).dev_hand[1]
        )
        for s in range(30)
    }
    assert len(hands) > 1  # the posterior is actually being sampled