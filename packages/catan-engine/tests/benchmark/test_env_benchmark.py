"""pytest-benchmark throughput benchmarks for the RL envs under random play.

Two rollouts driven by uniformly-random *legal* actions:

- ``BatchedCatanEnv`` -- a batch of games stepped in lockstep (the vectorised
  surface); one random legal action per lane per step.
- ``CatanAECEnv`` -- a single game, turn at a time (the PettingZoo surface).

Legality comes from the engine itself: the batched rollout samples via
``BatchedCatanEnv.random_actions`` and the AEC rollout via the action mask, so
the rollouts always make progress and exercise every action type (including the
forced DISCARD / MOVE_ROBBER after a 7). JIT compilation is warmed up before the
timed region.

The AEC surface needs the ``rl`` extra (``pettingzoo`` / ``gymnasium``).
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from catan_engine.env.aec import CatanAECEnv
from catan_engine.env.batched import BatchedCatanEnv


def _batched_rollout(seed: int, batch_size: int, steps: int) -> None:
    env = BatchedCatanEnv(batch_size=batch_size, seed=seed)
    key = jax.random.key(seed)
    for _ in range(steps):
        key, subkey = jax.random.split(key)
        action_type, params = env.random_actions(subkey)
        env.step(action_type, params)
    np.asarray(env.board[1].phase)  # force device->host sync so timing is honest


def _aec_rollout(seed: int, steps: int) -> None:
    e = CatanAECEnv(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        agent = e.agent_selection
        if e.terminations[agent] or e.truncations[agent]:
            e.reset(seed)
            agent = e.agent_selection
        legal = np.where(e.observe(agent)["action_mask"])[0]
        if legal.size == 0:  # only a terminal game has no legal move
            e.reset(seed)
            continue
        e.step(int(rng.choice(legal)))


def test_batched_env_random_rollout(benchmark: Any) -> None:
    """Throughput of a batch of games stepped with random legal actions."""
    batch_size, steps = 8, 40
    _batched_rollout(seed=0, batch_size=batch_size, steps=steps)  # warm up JIT
    benchmark(lambda: _batched_rollout(seed=0, batch_size=batch_size, steps=steps))


def test_aec_env_random_rollout(benchmark: Any) -> None:
    """Throughput of a single PettingZoo-AEC game under random legal play."""
    steps = 80
    _aec_rollout(seed=0, steps=steps)  # warm up JIT
    benchmark(lambda: _aec_rollout(seed=0, steps=steps))
