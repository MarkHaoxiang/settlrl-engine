"""Protocol-level tests, run against every shipped agent.

An agent in the ``POLICIES`` registry must pick a legal flat action whenever
one exists, be able to drive whole games in self-play, and be a pure function
of its inputs (same seed -> same trajectory). Each agent is exercised at a
player count it supports, through whichever protocol (observation / belief)
it declares.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from catan_engine.belief import BeliefView
from catan_engine.env import BatchedCatanEnv, Observation, flat_to_action

from catan_agents import POLICIES, AgentSpec, evaluate
from catan_agents.shared.policy import BeliefPolicy, Policy

BATCH = 8


def _acting_obs(env: BatchedCatanEnv) -> Observation:
    """Per-lane observation of that lane's acting player."""
    per_seat = [env.observe(i) for i in range(env.n_players)]
    lanes = jnp.arange(env.batch_size)
    return cast(
        Observation,
        jax.tree.map(lambda *xs: jnp.stack(xs)[env.agent_selection, lanes], *per_seat),
    )


def _acting_view(env: BatchedCatanEnv) -> BeliefView:
    """Per-lane ``BeliefView`` of that lane's acting player."""
    per_seat = [env.belief_view(i) for i in range(env.n_players)]
    lanes = jnp.arange(env.batch_size)
    return cast(
        BeliefView,
        jax.tree.map(lambda *xs: jnp.stack(xs)[env.agent_selection, lanes], *per_seat),
    )


def _self_play(
    spec: AgentSpec, seed: int, n_steps: int
) -> tuple[jax.Array, jax.Array]:
    """Drive ``n_steps`` of self-play; return the per-step ``(masks, actions)``."""
    env = BatchedCatanEnv(
        batch_size=BATCH,
        seed=seed,
        n_players=max(spec.n_players),
        track_beliefs=spec.observes == "belief",
    )
    act: Callable[[jax.Array], jax.Array]
    if spec.observes == "observation":
        obs_act = jax.jit(jax.vmap(cast(Policy, spec.policy)))
        act = lambda keys: obs_act(keys, _acting_obs(env), env.flat_mask())  # noqa: E731
    else:
        belief_act = jax.jit(jax.vmap(cast(BeliefPolicy, spec.policy)))
        act = lambda keys: belief_act(  # noqa: E731
            keys, env.board[0], _acting_view(env), env.agent_selection, env.flat_mask()
        )
    key = jax.random.key(seed)
    masks, actions = [], []
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        mask = env.flat_mask()
        flat = act(jax.random.split(k, BATCH))
        masks.append(mask)
        actions.append(flat)
        env.step(*flat_to_action(flat))
    return jnp.stack(masks), jnp.stack(actions)


@pytest.mark.parametrize("spec", POLICIES.values(), ids=POLICIES.keys())
def test_picks_only_legal_actions(spec: AgentSpec) -> None:
    masks, actions = _self_play(spec, seed=0, n_steps=150)
    # Whenever a lane has any legal move, the pick must be one of them.
    legal = jnp.take_along_axis(masks, actions[..., None], axis=2)[..., 0]
    assert bool(jnp.all(~masks.any(axis=2) | legal))


@pytest.mark.parametrize("spec", POLICIES.values(), ids=POLICIES.keys())
def test_same_seed_reproduces_rollout(spec: AgentSpec) -> None:
    _, first = _self_play(spec, seed=3, n_steps=60)
    _, second = _self_play(spec, seed=3, n_steps=60)
    assert bool(jnp.all(first == second))


@pytest.mark.parametrize("spec", POLICIES.values(), ids=POLICIES.keys())
def test_self_play_rollouts_complete_games(spec: AgentSpec) -> None:
    result = evaluate([spec, spec], n_steps=600, batch_size=BATCH, seed=0)
    assert result.wins.shape == (2,)
    assert result.episodes == int(result.wins.sum())
    assert result.episodes > 0
