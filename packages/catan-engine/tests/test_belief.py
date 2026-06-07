"""Soundness of belief tracking (``catan_engine.belief``) under random play.

The contract: everything in a ``BeliefState`` is derivable from public
information, the bounds always bracket the truth, and a censored state carries
nothing the observer couldn't know.
"""

import jax
import jax.numpy as jnp
import pytest

from catan_engine.belief import make_belief
from catan_engine.board.dev_cards import DEV_CARD_COUNTS
from catan_engine.env import BatchedCatanEnv

_DEV_COUNTS = jnp.asarray(DEV_CARD_COUNTS, jnp.int32)


def _rollout(env: BatchedCatanEnv, n_steps: int, seed: int = 42) -> None:
    key = jax.random.key(seed)
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_bounds_bracket_the_truth(n_players: int) -> None:
    env = BatchedCatanEnv(
        batch_size=8, seed=0, n_players=n_players, track_beliefs=True
    )
    key = jax.random.key(42)
    for _ in range(300):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        true = env.board[1].player_resources.astype(jnp.int32)  # (B, P, R)
        lo = env.beliefs.res_lo.astype(jnp.int32)  # (B, O, P, R)
        hi = env.beliefs.res_hi.astype(jnp.int32)
        assert bool(jnp.all(lo <= true[:, None]))
        assert bool(jnp.all(true[:, None] <= hi))
        # Each observer knows its own hand exactly.
        o = jnp.arange(n_players)
        assert bool(jnp.all(lo[:, o, o] == true))
        assert bool(jnp.all(hi[:, o, o] == true))


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_dev_pool_conservation(n_players: int) -> None:
    env = BatchedCatanEnv(
        batch_size=8, seed=1, n_players=n_players, track_beliefs=True
    )
    _rollout(env, 300)
    state = env.board[1]
    held = state.dev_hand.astype(jnp.int32).sum(axis=1)
    played = env.beliefs.dev_played.astype(jnp.int32)
    assert bool(
        jnp.all(state.dev_deck.astype(jnp.int32) + held + played == _DEV_COUNTS)
    )


def test_two_player_beliefs_are_exact() -> None:
    # With two players every flow is mutually visible (a steal involves both
    # seats) and public totals pin the rest: the bounds must stay tight.
    env = BatchedCatanEnv(batch_size=8, seed=2, n_players=2, track_beliefs=True)
    key = jax.random.key(7)
    for _ in range(300):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))
        assert bool(jnp.all(env.beliefs.res_lo == env.beliefs.res_hi))


def test_third_party_steals_create_uncertainty() -> None:
    # The converse: with 4 players, random play eventually produces a steal
    # some observer didn't take part in, and their bounds must open up.
    env = BatchedCatanEnv(batch_size=8, seed=3, n_players=4, track_beliefs=True)
    _rollout(env, 400)
    slack = (
        env.beliefs.res_hi.astype(jnp.int32) - env.beliefs.res_lo.astype(jnp.int32)
    )
    assert int(slack.sum()) > 0


def test_censor_hides_everything_hidden() -> None:
    env = BatchedCatanEnv(batch_size=4, seed=4, n_players=4, track_beliefs=True)
    _rollout(env, 300)
    state = env.board[1]
    for me in range(4):
        censored, pb = env.belief_view(me)
        others = [p for p in range(4) if p != me]
        # Own rows exact; opponents' dev hands gone; resources at the proven floor.
        assert bool(
            jnp.all(censored.player_resources[:, me] == state.player_resources[:, me])
        )
        assert bool(jnp.all(censored.dev_hand[:, me] == state.dev_hand[:, me]))
        assert bool(jnp.all(censored.dev_hand[:, others] == 0))
        assert bool(
            jnp.all(censored.player_resources[:, others] == env.beliefs.res_lo[:, me, others])
        )
        # The censored deck is the observer's unseen pool.
        pool = (
            _DEV_COUNTS
            - env.beliefs.dev_played.astype(jnp.int32)
            - state.dev_hand[:, me].astype(jnp.int32)
        )
        assert bool(jnp.all(censored.dev_deck.astype(jnp.int32) == pool))
        # The PRNG key carries no future randomness.
        assert bool(
            jnp.all(
                jax.random.key_data(censored.key)
                == jax.random.key_data(jax.random.key(0))
            )
        )
        # The public counts are the truth's.
        assert bool(
            jnp.all(pb.hand_size == state.player_resources.astype(jnp.int32).sum(axis=2))
        )
        assert bool(
            jnp.all(pb.dev_count == state.dev_hand.astype(jnp.int32).sum(axis=2))
        )
        assert bool(
            jnp.all(pb.res_total == state.player_resources.astype(jnp.int32).sum(axis=1))
        )


def test_belief_resets_with_auto_reset() -> None:
    # Run long enough for lanes to finish (2p games end quickly): beliefs of
    # replaced lanes must restart from the empty-board belief, staying sound.
    env = BatchedCatanEnv(batch_size=4, seed=5, n_players=2, track_beliefs=True)
    _rollout(env, 4_000)
    true = env.board[1].player_resources.astype(jnp.int32)
    assert bool(jnp.all(env.beliefs.res_lo.astype(jnp.int32) <= true[:, None]))
    assert bool(jnp.all(true[:, None] <= env.beliefs.res_hi.astype(jnp.int32)))


def test_tracking_off_raises() -> None:
    env = BatchedCatanEnv(batch_size=1, seed=0, n_players=2)
    with pytest.raises(RuntimeError, match="track_beliefs"):
        env.beliefs


def test_make_belief_shapes() -> None:
    b = make_belief(batch_size=3, n_players=3)
    assert b.res_lo.shape == (3, 3, 3, 5)
    assert b.res_hi.shape == (3, 3, 3, 5)
    assert b.dev_played.shape == (3, 5)
