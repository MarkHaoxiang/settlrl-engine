"""Greedy over a value function: pick the action whose successor scores best."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, IntScalar
from catan_engine.env import N_FLAT, flat_to_action
from catan_engine.mechanics.action import apply_action

from catan_agents.shared.policy import FlatAction, FlatMask, StatePolicy
from catan_agents.shared.value import ValueFunction, heuristic_value
from catan_agents.two_player.belief import redeal_dev_cards

# Static decode of every flat row: its action type and (idx, target) params.
_ROW_TYPE, _ROW_PARAMS = flat_to_action(jnp.arange(N_FLAT))


def make_greedy(value: ValueFunction) -> StatePolicy:
    """One-step lookahead: apply every legal action and argmax ``value``.

    The state is determinized from ``key`` first: its PRNG key is replaced
    (stochastic successors — dice, steals, dev-card draws — are the policy's
    own samples, not a preview of the environment's outcomes) and the
    opponent's hidden dev-card identities are re-dealt from the player's
    unseen pool.
    """

    def policy(
        key: jax.Array,
        layout: BoardLayout,
        state: BoardState,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction:
        k_state, k_deal, k_noise = jax.random.split(key, 3)
        state = redeal_dev_cards(k_deal, state._replace(key=k_state), player)
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
