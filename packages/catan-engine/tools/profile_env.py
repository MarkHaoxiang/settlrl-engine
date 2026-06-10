"""Profiler for the batched Catan environment's random-rollout loop.

Answers "where does the wall-clock go when stepping ``BatchedCatanEnv`` under
random play?". The default mode runs the rollout under :mod:`cProfile` and
prints the hottest frames -- which is what actually localises the cost (e.g. a
hidden per-step device->host sync shows up as time in ``jax...array._value``,
something a coarse wall-clock timer would silently fold into ``step``).

A step has two parts: ``random_actions(key)`` (sample a legal action per lane --
this enumerates the whole flat action table, ``N_FLAT`` ~= 556 moves, against
every lane) and ``step(action_type, params)`` (apply it). cProfile attributes
time to whichever is actually expensive.

``--trace DIR`` instead writes a JAX/XLA device trace to ``DIR`` for the on-device
op breakdown (open with ``tensorboard --logdir DIR``); use it once cProfile has
pointed at the host side and you want to see the device side.

Usage (from the repo root)::

    uv run --package catan-engine python packages/catan-engine/tools/profile_env.py
    uv run --package catan-engine python packages/catan-engine/tools/profile_env.py \
        --batch-size 100 --steps 40
    uv run --package catan-engine python packages/catan-engine/tools/profile_env.py \
        --batch-size 100 --trace /tmp/catan-trace
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from time import perf_counter

import jax
from catan_engine.env.batched import N_FLAT, BatchedCatanEnv


def _rollout(env: BatchedCatanEnv, key: jax.Array, steps: int) -> jax.Array:
    """Run ``steps`` random-action steps, returning the final ``key`` (lazy)."""
    for _ in range(steps):
        key, subkey = jax.random.split(key)
        action_type, params = env.random_actions(subkey)
        env.step(action_type, params)
    return key


def _warmup(env: BatchedCatanEnv, key: jax.Array, steps: int = 2) -> jax.Array:
    """Trigger JIT compilation for this batch shape before profiling."""
    key = _rollout(env, key, steps)
    jax.block_until_ready(env.board[1])  # type: ignore[no-untyped-call]
    return key


def run_cprofile(batch_size: int, steps: int, seed: int, top: int) -> None:
    env = BatchedCatanEnv(batch_size=batch_size, seed=seed)
    key = _warmup(env, jax.random.key(seed))

    pr = cProfile.Profile()
    pr.enable()
    t0 = perf_counter()
    key = _rollout(env, key, steps)
    jax.block_until_ready(env.board[1])  # type: ignore[no-untyped-call]
    wall = perf_counter() - t0
    pr.disable()

    print(
        f"\nbatch_size={batch_size}  steps={steps}  N_FLAT={N_FLAT}\n"
        f"wall {wall:.3f}s  ->  {wall / steps * 1e3:.1f} ms/step, "
        f"{batch_size * steps / wall:,.0f} env-steps/s\n"
    )
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(top)
    print(f"top {top} frames by cumulative time:\n")
    print(s.getvalue())


def run_trace(batch_size: int, steps: int, seed: int, trace_dir: str) -> None:
    env = BatchedCatanEnv(batch_size=batch_size, seed=seed)
    key = _warmup(env, jax.random.key(seed))
    with jax.profiler.trace(trace_dir):
        key = _rollout(env, key, steps)
        jax.block_until_ready(env.board[1])  # type: ignore[no-untyped-call]
    print(
        f"\nDevice trace written to {trace_dir!r}.\n"
        f"View it with:  tensorboard --logdir {trace_dir}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch-size", type=int, default=100, help="batch size to profile")
    p.add_argument("--steps", type=int, default=40, help="steps per rollout")
    p.add_argument("--seed", type=int, default=0, help="PRNG / board seed")
    p.add_argument("--top", type=int, default=20, help="cProfile frames to show")
    p.add_argument(
        "--trace",
        metavar="DIR",
        help="write a JAX/XLA device trace to DIR instead of running cProfile",
    )
    args = p.parse_args()

    if args.trace:
        run_trace(args.batch_size, args.steps, args.seed, args.trace)
    else:
        run_cprofile(args.batch_size, args.steps, args.seed, args.top)


if __name__ == "__main__":
    main()
