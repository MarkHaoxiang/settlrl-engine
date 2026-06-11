"""Greedy over a value function: pick the action whose successor scores best."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from catan_engine.belief import BeliefView
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import IntScalar, KeyScalar
from catan_engine.env import N_FLAT, flat_to_action
from catan_engine.mechanics.action import apply_action

from catan_agents.shared.policy import BeliefPolicy, FlatAction, FlatMask
from catan_agents.shared.sample import sample_world
from catan_agents.shared.value import ValueFunction, heuristic_value

# Static decode of every flat row: its action type and (idx, target) params.
_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))


def make_greedy(value: ValueFunction) -> BeliefPolicy:
    """One-step lookahead: apply every legal action and argmax ``value``.

    The view is first made concrete with one
    :func:`~catan_agents.shared.sample.sample_world` draw, so stochastic
    successors — dice, steals, dev-card draws — are the policy's own samples
    over a world consistent with what the player knows.
    """

    def policy(
        key: KeyScalar,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        k_world, k_noise = jax.random.split(key)
        state = sample_world(k_world, view, player)
        successors, _ = jax.vmap(apply_action, in_axes=(None, None, 0, 0, 0))(
            layout, state, _ROW_TYPE, _ROW_PARAMS, mask
        )
        values = jax.vmap(value, in_axes=(None, 0, None))(layout, successors, player)
        # Tiny uniform noise breaks exact-value ties at random.
        noise = jax.random.uniform(k_noise, (N_FLAT,)) * 1e-4
        return jnp.argmax(jnp.where(mask, values + noise, -jnp.inf))

    return policy


lookahead_policy = make_greedy(heuristic_value)
"""The value-greedy agent: one-step lookahead over :func:`heuristic_value`."""
