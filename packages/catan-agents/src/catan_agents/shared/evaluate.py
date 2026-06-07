"""Head-to-head evaluation: seat policies in a batch of games and count wins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from catan_engine.env import BatchedCatanEnv, flat_to_action

from catan_agents.policy import Policy


class EvalResult(NamedTuple):
    """Outcome of an :func:`evaluate` run."""

    wins: jax.Array
    """``(n_seats,)`` games won by each seat."""

    episodes: int
    """Total games completed within the step budget."""


def evaluate(
    policies: Sequence[Policy],
    *,
    n_steps: int,
    batch_size: int = 64,
    seed: int = 0,
    number_placement: Literal["random", "spiral"] = "random",
) -> EvalResult:
    """Play ``len(policies)`` seats (2..4 players) for ``n_steps`` env steps.

    Each policy occupies one seat in every game of the batch; finished games
    auto-reset, so ``episodes`` counts every game completed within the budget
    (games still running at the end are discarded). Deterministic for a given
    configuration and ``seed``.
    """
    n = len(policies)
    env = BatchedCatanEnv(
        batch_size=batch_size,
        seed=seed,
        reward="sparse",
        n_players=n,
        number_placement=number_placement,
    )
    seats = [jax.jit(jax.vmap(p)) for p in policies]
    lanes = jnp.arange(batch_size)
    key = jax.random.key(seed)
    wins = jnp.zeros((n,), jnp.float32)
    for _ in range(n_steps):
        key, *seat_keys = jax.random.split(key, n + 1)
        mask = env.flat_mask()
        # Every seat picks a move in every lane; the acting seat's is kept.
        picks = jnp.stack(
            [
                seat(jax.random.split(k, batch_size), env.observe(i), mask)
                for i, (seat, k) in enumerate(zip(seats, seat_keys))
            ]
        )
        flat = picks[env.agent_selection, lanes]
        env.step(*flat_to_action(flat))
        wins = wins + env.rewards.sum(axis=0)
    return EvalResult(wins=wins, episodes=int(wins.sum()))
