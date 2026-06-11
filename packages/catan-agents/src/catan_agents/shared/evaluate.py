"""Head-to-head evaluation: seat agents in a batch of games and count wins."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from catan_engine.env import BatchedCatanEnv, flat_to_action

from catan_agents.shared.policy import AgentSpec, BeliefPolicy, Policy


class EvalResult(NamedTuple):
    """Outcome of an :func:`evaluate` run."""

    wins: jax.Array
    """``(n_seats,)`` games won by each seat."""

    episodes: int
    """Total games completed within the step budget."""


Seat = Callable[[jax.Array, BatchedCatanEnv, int], jax.Array]
"""Internal: a seated agent picking a ``(B,)`` flat action for every lane."""


def _seat(agent: AgentSpec | Policy, n: int, i: int) -> Seat:
    """Wrap one agent as a vmapped per-lane picker for seat ``i`` of ``n``."""
    spec = (
        agent
        if isinstance(agent, AgentSpec)
        else AgentSpec(lambda: agent, "observation", frozenset((2, 3, 4)))
    )
    if n not in spec.n_players:
        raise ValueError(f"seat {i} does not support {n}-player games")
    if spec.observes == "observation":
        obs_act = jax.jit(jax.vmap(cast(Policy, spec.policy)))
        return lambda keys, env, seat: obs_act(keys, env.observe(seat), env.flat_mask())
    belief_act = jax.jit(
        jax.vmap(cast(BeliefPolicy, spec.policy), in_axes=(0, 0, 0, None, 0))
    )
    return lambda keys, env, seat: belief_act(
        keys, env.board[0], env.belief_view(seat), jnp.int32(seat), env.flat_mask()
    )


# Step cap per requested episode in n_episodes mode, guarding against agents
# that never finish a game (a full game is well under this many steps).
_MAX_STEPS_PER_EPISODE = 5_000


def evaluate(
    agents: Sequence[AgentSpec | Policy],
    *,
    n_steps: int | None = None,
    n_episodes: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
    number_placement: Literal["random", "spiral"] = "random",
) -> EvalResult:
    """Play ``len(agents)`` seats (2..4 players) under exactly one budget:
    ``n_steps`` env steps, or until at least ``n_episodes`` games finish.

    Each agent (an :class:`AgentSpec`, or a bare :class:`Policy` valid at any
    count) occupies one seat in every game of the batch; finished games
    auto-reset, so ``episodes`` counts every game completed within the budget
    (games still running at the end are discarded; several lanes can finish on
    the same step, so ``n_episodes`` may be overshot). Deterministic for a
    given configuration and ``seed``.
    """
    if (n_steps is None) == (n_episodes is None):
        raise ValueError("provide exactly one of n_steps / n_episodes")
    n = len(agents)
    env = BatchedCatanEnv(
        batch_size=batch_size,
        seed=seed,
        reward="sparse",
        n_players=n,
        number_placement=number_placement,
        track_beliefs=any(
            isinstance(a, AgentSpec) and a.observes == "belief" for a in agents
        ),
    )
    seats = [_seat(agent, n, i) for i, agent in enumerate(agents)]
    lanes = jnp.arange(batch_size)
    key = jax.random.key(seed)
    wins = jnp.zeros((n,), jnp.float32)
    if n_steps is None:
        assert n_episodes is not None
        n_steps = _MAX_STEPS_PER_EPISODE * (n_episodes // batch_size + 1)
    for _ in range(n_steps):
        key, *seat_keys = jax.random.split(key, n + 1)
        # Every seat picks a move in every lane; the acting seat's is kept.
        picks = jnp.stack(
            [
                seat(jax.random.split(k, batch_size), env, i)
                for i, (seat, k) in enumerate(zip(seats, seat_keys, strict=True))
            ]
        )
        flat = picks[env.agent_selection, lanes]
        env.step(*flat_to_action(flat))
        wins = wins + env.rewards.sum(axis=0)
        # The episode budget syncs on the win count each step; the step budget
        # stays sync-free.
        if n_episodes is not None and int(wins.sum()) >= n_episodes:
            break
    return EvalResult(wins=wins, episodes=int(wins.sum()))
