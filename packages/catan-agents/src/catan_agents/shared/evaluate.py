"""Head-to-head evaluation: seat agents in a batch of games and count wins."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from catan_engine.belief import BeliefState, belief_view
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, KeyScalar
from catan_engine.env import ActionParams, BatchedCatanEnv, flat_to_action, observe_for
from catan_engine.env.batched import Actor, AgentSelectionArray
from catan_engine.mechanics.flat import FlatMaskArray
from jaxtyping import Array, Float, Int

from catan_agents.shared.policy import AgentSpec, BeliefSpec, ObservationSpec, Policy


class EvalResult(NamedTuple):
    """Outcome of an :func:`evaluate` run."""

    wins: Float[Array, "seats"]
    """Games won by each seat."""

    episodes: int
    """Total games completed within the step budget."""


_Picker = Callable[
    [KeyScalar, BoardLayout, BoardState, BeliefState | None, FlatMaskArray],
    Int[Array, "batch"],
]
"""Internal: one seat's per-lane flat-action picks, traceable inside a scan."""


def _picker(agent: ObservationSpec | BeliefSpec | Policy, n: int, i: int) -> _Picker:
    """Wrap one agent as seat ``i`` of ``n``."""
    spec = (
        agent
        if isinstance(agent, AgentSpec)
        else ObservationSpec(lambda: agent, frozenset((2, 3, 4)))
    )
    if n not in spec.n_players:
        raise ValueError(f"seat {i} does not support {n}-player games")
    if isinstance(spec, ObservationSpec):
        obs_act = jax.vmap(spec.policy)

        def pick_obs(
            key: KeyScalar,
            layout: BoardLayout,
            state: BoardState,
            belief: BeliefState | None,
            avail: FlatMaskArray,
        ) -> Int[Array, "batch"]:
            b = avail.shape[0]
            obs = observe_for(layout, state, jnp.full((b,), i, jnp.int32))
            return obs_act(jax.random.split(key, b), obs, avail)

        return pick_obs
    belief_act = jax.vmap(spec.policy, in_axes=(0, 0, 0, None, 0))

    def pick_belief(
        key: KeyScalar,
        layout: BoardLayout,
        state: BoardState,
        belief: BeliefState | None,
        avail: FlatMaskArray,
    ) -> Int[Array, "batch"]:
        assert belief is not None  # evaluate() tracks beliefs for BeliefSpec seats
        b = avail.shape[0]
        view = jax.vmap(belief_view, in_axes=(0, 0, None))(state, belief, i)
        return belief_act(jax.random.split(key, b), layout, view, jnp.int32(i), avail)

    return pick_belief


def _actor(pickers: Sequence[_Picker]) -> Actor:
    """Every seat picks a move in every lane; the acting seat's is kept."""

    def actor(
        key: KeyScalar,
        layout: BoardLayout,
        state: BoardState,
        belief: BeliefState | None,
        avail: FlatMaskArray,
        agent_sel: AgentSelectionArray,
    ) -> tuple[jax.Array, ActionParams]:
        keys = jax.random.split(key, len(pickers))
        picks = jnp.stack(
            [
                pick(k, layout, state, belief, avail)
                for pick, k in zip(pickers, keys, strict=True)
            ]
        )
        flat = picks[agent_sel, jnp.arange(avail.shape[0])]
        return flat_to_action(flat)

    return actor


# Step cap per requested episode in n_episodes mode, guarding against agents
# that never finish a game (a full game is well under this many steps).
_MAX_STEPS_PER_EPISODE = 5_000

# Steps fused into one rollout scan between win-count syncs: long enough to
# amortise the dispatch, short enough to keep the n_episodes overshoot small.
_SYNC_WINDOW = 64


def evaluate(
    agents: Sequence[ObservationSpec | BeliefSpec | Policy],
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
    (games still running at the end are discarded; lanes finishing within the
    same sync window can overshoot ``n_episodes``). Deterministic for a given
    configuration and ``seed``.
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
        track_beliefs=any(isinstance(a, BeliefSpec) for a in agents),
    )
    actor = _actor([_picker(agent, n, i) for i, agent in enumerate(agents)])
    key = jax.random.key(seed)
    wins = jnp.zeros((n,), jnp.float32)
    total = (
        n_steps
        if n_steps is not None
        else _MAX_STEPS_PER_EPISODE * ((n_episodes or 0) // batch_size + 1)
    )
    done = 0
    while done < total:
        window = min(_SYNC_WINDOW, total - done)
        key, k = jax.random.split(key)
        # One fused scan per window; the win count only syncs between windows.
        wins = wins + env.rollout(k, window, actor=actor).sum(axis=0)
        done += window
        if n_episodes is not None and int(wins.sum()) >= n_episodes:
            break
    return EvalResult(wins=wins, episodes=int(wins.sum()))
