"""Soundness of belief tracking (``catan_engine.belief``) under random play.

The contract: everything in a ``BeliefState`` is derivable from public
information, the bounds always bracket the truth, and a ``BeliefView`` carries
nothing the observer couldn't know — structurally, since its type has no field
for anything hidden.
"""

import jax
import jax.numpy as jnp
import pytest
from catan_engine.belief import PublicState
from catan_engine.board.dev_cards import DEV_CARD_COUNTS
from catan_engine.board.state import BoardState
from catan_engine.env import BatchedCatanEnv

_DEV_COUNTS = jnp.asarray(DEV_CARD_COUNTS, jnp.int32)


def _rollout(env: BatchedCatanEnv, n_steps: int, seed: int = 42) -> None:
    key = jax.random.key(seed)
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        env.step(*env.random_actions(k))


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_bounds_bracket_the_truth(n_players: int) -> None:
    env = BatchedCatanEnv(batch_size=8, seed=0, n_players=n_players, track_beliefs=True)
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
    env = BatchedCatanEnv(batch_size=8, seed=1, n_players=n_players, track_beliefs=True)
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
    slack = env.beliefs.res_hi.astype(jnp.int32) - env.beliefs.res_lo.astype(jnp.int32)
    assert int(slack.sum()) > 0


_HIDDEN_FIELDS = {"player_resources", "dev_hand", "dev_deck", "dev_bought", "key"}


def test_every_board_field_is_classified() -> None:
    # PublicState plus hidden must cover BoardState exactly: a new BoardState
    # field can't slip into (or out of) the view layer unclassified.
    assert set(PublicState._fields) | _HIDDEN_FIELDS == set(BoardState._fields)
    assert set(PublicState._fields) & _HIDDEN_FIELDS == set()


def test_belief_view_carries_nothing_hidden() -> None:
    env = BatchedCatanEnv(batch_size=4, seed=4, n_players=4, track_beliefs=True)
    _rollout(env, 300)
    state = env.board[1]
    for me in range(4):
        view = env.belief_view(me)
        # Structural: the view's type has no field for anything hidden.
        for name in _HIDDEN_FIELDS:
            assert not hasattr(view, name) and not hasattr(view.public, name)
        # The public fields are the truth's, field for field.
        for name in PublicState._fields:
            assert bool(jnp.all(getattr(view.public, name) == getattr(state, name)))
        # Own knowledge is exact; bounds are the observer's slice.
        assert bool(jnp.all(view.own_dev == state.dev_hand[:, me]))
        assert bool(jnp.all(view.belief.res_lo == env.beliefs.res_lo[:, me]))
        assert bool(jnp.all(view.belief.res_hi == env.beliefs.res_hi[:, me]))
        # Own purchases survive only on the observer's turn.
        own_turn = state.current_player.astype(jnp.int32) == me
        assert bool(
            jnp.all(
                view.own_bought == jnp.where(own_turn[:, None], state.dev_bought, 0)
            )
        )
        # The unseen dev pool by conservation.
        pool = (
            _DEV_COUNTS
            - env.beliefs.dev_played.astype(jnp.int32)
            - state.dev_hand[:, me].astype(jnp.int32)
        )
        assert bool(jnp.all(view.unseen_dev.astype(jnp.int32) == pool))
        # The public counts are the truth's.
        pb = view.belief
        assert bool(
            jnp.all(
                pb.hand_size == state.player_resources.astype(jnp.int32).sum(axis=2)
            )
        )
        assert bool(
            jnp.all(pb.dev_count == state.dev_hand.astype(jnp.int32).sum(axis=2))
        )
        assert bool(
            jnp.all(
                pb.res_total == state.player_resources.astype(jnp.int32).sum(axis=1)
            )
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
        _ = env.beliefs
