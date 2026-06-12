"""pytest-benchmark latency/throughput benchmarks for the shipped agents.

Three measures, each over every agent in ``POLICIES`` at its shipped
(default) parameters:

- ``test_agent_move`` -- steady-state latency of one jitted decision on a
  fixed mid-game position, swept over batch sizes (1 / 32) to show how moves
  amortise across lanes.
- ``test_sample_world`` -- the determinization primitive: one ``BeliefView``
  per lane filled into a playable world.
- ``test_selfplay_window`` -- the ``evaluate`` hot loop: one fused
  ``rollout(actor=...)`` scan of self-play, every seat picking each step.

All run 2-player games (search cost is near-identical at 4) and are swept
over devices (CPU always; CUDA when a GPU-enabled jaxlib sees a device,
otherwise the CUDA variants skip), pinned via ``jax.default_device``. JIT is
warmed up before each timed region.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from catan_agents import (
    POLICIES,
    BeliefSpec,
    ObservationSpec,
    StatefulSpec,
    sample_world,
)
from catan_agents.evaluate import _actor, _picker
from catan_engine.belief import BeliefView
from catan_engine.env import BatchedCatanEnv, Observation


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

_PLAYERS = 2
_MIDGAME_STEPS = 150


@functools.cache
def _midgame_env(batch_size: int, device: str) -> BatchedCatanEnv:
    """A batch of mid-game positions (random play), reused across benchmarks
    that only read it."""
    with jax.default_device(jax.devices(device)[0]):
        env = BatchedCatanEnv(
            batch_size=batch_size, seed=0, n_players=_PLAYERS, track_beliefs=True
        )
        env.rollout(jax.random.key(0), _MIDGAME_STEPS)
    return env


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


def _move(
    spec: ObservationSpec | BeliefSpec | StatefulSpec, env: BatchedCatanEnv
) -> Callable[[], jax.Array]:
    """One batched decision on the env's current position (jitted for the
    pure policies; the per-lane Python loop a stateful seat actually costs)."""
    keys = jax.random.split(jax.random.key(1), env.batch_size)
    mask = env.flat_mask()
    if isinstance(spec, StatefulSpec):
        agents = [spec.policy(lane) for lane in range(env.batch_size)]
        obs = _acting_obs(env)

        def stateful() -> jax.Array:
            # The fetch is in the timed region: the stepwise driver pays it.
            obs_h = cast("dict[str, np.ndarray]", jax.device_get(obs))
            mask_h = np.asarray(mask)
            picks = [
                agents[lane].act({k: v[lane] for k, v in obs_h.items()}, mask_h[lane])
                for lane in range(env.batch_size)
            ]
            return jnp.asarray(picks, jnp.int32)

        return stateful
    if isinstance(spec, ObservationSpec):
        obs_act = jax.jit(jax.vmap(spec.policy))
        obs = _acting_obs(env)
        return lambda: obs_act(keys, obs, mask)
    belief_act = jax.jit(jax.vmap(spec.policy))
    layout, view, player = env.board[0], _acting_view(env), env.agent_selection
    return lambda: belief_act(keys, layout, view, player, mask)


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 32], ids=lambda b: f"B{b}")
@pytest.mark.parametrize("name", sorted(POLICIES))
def test_agent_move(benchmark: Any, name: str, batch_size: int, device: str) -> None:
    """Steady-state latency of one decision on a fixed mid-game position."""
    benchmark.group = f"agent_move[B{batch_size}-{device}]"
    with jax.default_device(jax.devices(device)[0]):
        move = _move(POLICIES[name], _midgame_env(batch_size, device))
        np.asarray(move())  # warm up JIT; sync for honest timing
        benchmark(lambda: np.asarray(move()))


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 64], ids=lambda b: f"B{b}")
def test_sample_world(benchmark: Any, batch_size: int, device: str) -> None:
    """Latency of determinizing one ``BeliefView`` per lane into a playable
    world (the root step of every search agent)."""
    benchmark.group = f"sample_world[{device}]"
    with jax.default_device(jax.devices(device)[0]):
        env = _midgame_env(batch_size, device)
        view, player = _acting_view(env), env.agent_selection
        keys = jax.random.split(jax.random.key(1), batch_size)
        sample = jax.jit(jax.vmap(sample_world))
        # .phase: device->host sync of one output leaf for honest timing.
        np.asarray(sample(keys, view, player).phase)  # warm up JIT
        benchmark(lambda: np.asarray(sample(keys, view, player).phase))


_WINDOW = 32
_EVAL_BATCH = 32


@pytest.mark.benchmark
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", sorted(POLICIES))
def test_selfplay_window(benchmark: Any, name: str, device: str) -> None:
    """Throughput of one fused self-play window through the engine's
    ``rollout(actor=...)`` seam -- the ``evaluate`` hot loop (every seat picks
    in every lane each step; lanes auto-reset, so rounds stay mid-game)."""
    benchmark.group = f"selfplay_window[{device}]"
    spec = POLICIES[name]
    if isinstance(spec, StatefulSpec):
        pytest.skip("stateful seats run the per-step Python driver, not the scan")
    with jax.default_device(jax.devices(device)[0]):
        env = BatchedCatanEnv(
            batch_size=_EVAL_BATCH,
            seed=0,
            n_players=_PLAYERS,
            track_beliefs=isinstance(spec, BeliefSpec),
        )
        actor = _actor([_picker(spec, _PLAYERS, i) for i in range(_PLAYERS)])
        key = jax.random.key(0)
        np.asarray(env.rollout(key, _WINDOW, actor=actor))  # warm up JIT
        benchmark(lambda: np.asarray(env.rollout(key, _WINDOW, actor=actor)))
