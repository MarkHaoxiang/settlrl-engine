"""pytest-benchmark throughput benchmarks for the RL envs under random play.

Two rollouts driven by uniformly-random *legal* actions:

- ``BatchedCatanEnv`` -- a batch of games stepped in lockstep (the vectorised
  surface); one random legal action per lane per step. Swept over batch sizes
  (1 / 8 / 64 / 512) to show how the per-step cost amortises across the batch.
- ``CatanAECEnv`` -- a single game, turn at a time (the PettingZoo surface).

Both are swept over 2- and 4-player boards, and over devices (CPU always; CUDA
only when a GPU-enabled jaxlib sees a device — install the ``cuda`` extra —
otherwise the CUDA variants skip). The device is pinned explicitly via
``jax.default_device`` so the CPU numbers stay CPU numbers even when a GPU is
the default backend.

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
import pytest

from catan_engine.env.aec import CatanAECEnv
from catan_engine.env.batched import BatchedCatanEnv


def _cuda_available() -> bool:
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:  # no CUDA plugin installed, or plugin sees no GPU
        return False


# Device sweep: CPU always; CUDA only when available (install the `cuda` extra).
_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not _cuda_available(),
            reason="no CUDA device (needs the `cuda` extra and an NVIDIA GPU)",
        ),
    ),
]


def _batched_rollout(
    seed: int, batch_size: int, n_players: int, steps: int, device: str
) -> None:
    with jax.default_device(jax.devices(device)[0]):
        env = BatchedCatanEnv(batch_size=batch_size, n_players=n_players, seed=seed)
        key = jax.random.key(seed)
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            action_type, params = env.random_actions(subkey)
            env.step(action_type, params)
        np.asarray(env.board[1].phase)  # force device->host sync, honest timing


def _aec_rollout(seed: int, n_players: int, steps: int, device: str) -> None:
    with jax.default_device(jax.devices(device)[0]):
        e = CatanAECEnv(n_players=n_players, seed=seed)
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


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("n_players", [2, 4], ids=lambda n: f"{n}p")
@pytest.mark.parametrize("batch_size", [1, 8, 64, 512])
def test_batched_env_random_rollout(
    benchmark: Any, batch_size: int, n_players: int, device: str
) -> None:
    """Throughput of a batch of games stepped with random legal actions.

    Swept across batch sizes to show how the vectorised surface amortises the
    per-step cost: wall-clock for a fixed number of steps grows far slower than
    the batch (more games stepped per unit time)."""
    steps = 500
    benchmark.group = f"batched_env_random_rollout[{n_players}p-{device}]"
    _batched_rollout(  # warm up JIT
        seed=0, batch_size=batch_size, n_players=n_players, steps=steps, device=device
    )
    benchmark(
        lambda: _batched_rollout(
            seed=0,
            batch_size=batch_size,
            n_players=n_players,
            steps=steps,
            device=device,
        )
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("n_players", [2, 4], ids=lambda n: f"{n}p")
def test_aec_env_random_rollout(benchmark: Any, n_players: int, device: str) -> None:
    """Throughput of a single PettingZoo-AEC game under random legal play."""
    steps = 500
    benchmark.group = f"aec_env_random_rollout[{device}]"
    _aec_rollout(seed=0, n_players=n_players, steps=steps, device=device)  # warm up
    benchmark(
        lambda: _aec_rollout(seed=0, n_players=n_players, steps=steps, device=device)
    )
