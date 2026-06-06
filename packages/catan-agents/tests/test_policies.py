"""Protocol-level tests, run against every shipped policy.

A policy that satisfies :class:`catan_agents.Policy` must pick a legal flat
action whenever one exists, be able to drive whole games in self-play, and be
a pure function of its inputs (same seed -> same trajectory). Register new
policies in ``POLICIES`` to put them under the same tests.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from catan_engine.env import BatchedCatanEnv, Observation, flat_to_action

from catan_agents import evaluate, greedy_policy, random_policy
from catan_agents.policy import Policy

POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "greedy": greedy_policy,
}

BATCH = 8


def _acting_obs(env: BatchedCatanEnv) -> Observation:
    """Per-lane observation of that lane's acting player."""
    per_seat = [env.observe(i) for i in range(env.n_players)]
    lanes = jnp.arange(env.batch_size)
    return cast(
        Observation,
        jax.tree.map(lambda *xs: jnp.stack(xs)[env.agent_selection, lanes], *per_seat),
    )


def _self_play(
    policy: Policy, seed: int, n_steps: int
) -> tuple[jax.Array, jax.Array]:
    """Drive ``n_steps`` of self-play; return the per-step ``(masks, actions)``."""
    env = BatchedCatanEnv(batch_size=BATCH, seed=seed)
    act = jax.jit(jax.vmap(policy))
    key = jax.random.key(seed)
    masks, actions = [], []
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        mask = env.flat_mask()
        flat = act(jax.random.split(k, BATCH), _acting_obs(env), mask)
        masks.append(mask)
        actions.append(flat)
        env.step(*flat_to_action(flat))
    return jnp.stack(masks), jnp.stack(actions)


@pytest.mark.parametrize("policy", POLICIES.values(), ids=POLICIES.keys())
def test_picks_only_legal_actions(policy: Policy) -> None:
    masks, actions = _self_play(policy, seed=0, n_steps=150)
    # Whenever a lane has any legal move, the pick must be one of them.
    legal = jnp.take_along_axis(masks, actions[..., None], axis=2)[..., 0]
    assert bool(jnp.all(~masks.any(axis=2) | legal))


@pytest.mark.parametrize("policy", POLICIES.values(), ids=POLICIES.keys())
def test_same_seed_reproduces_rollout(policy: Policy) -> None:
    _, first = _self_play(policy, seed=3, n_steps=60)
    _, second = _self_play(policy, seed=3, n_steps=60)
    assert bool(jnp.all(first == second))


@pytest.mark.parametrize("policy", POLICIES.values(), ids=POLICIES.keys())
def test_self_play_rollouts_complete_games(policy: Policy) -> None:
    result = evaluate([policy, policy], n_steps=600, batch_size=BATCH, seed=0)
    assert result.wins.shape == (2,)
    assert result.episodes == int(result.wins.sum())
    assert result.episodes > 0
