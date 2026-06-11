"""Baseline policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from catan_engine.board.state import KeyScalar
from catan_engine.env import N_ACTION_TYPES, N_FLAT, Observation, flat_to_action

from catan_agents.shared.policy import FlatAction, FlatMask

_ROW_TYPE, _ = flat_to_action(jnp.arange(N_FLAT))


def random_policy(key: KeyScalar, obs: Observation, mask: FlatMask) -> FlatAction:
    """Uniform over the legal action *types*, then uniform over that type's
    legal moves (``obs`` is ignored).

    Two-stage rather than flat-uniform so bulk-enumerated types don't crowd
    out single-row ones: the ~100 trade-proposal rows would otherwise make
    almost every MAIN move an offer (each costing an extra response step),
    stretching random games several-fold.
    """
    k_type, k_row = jax.random.split(key)
    type_legal = jnp.zeros((N_ACTION_TYPES,), bool).at[_ROW_TYPE].max(mask)
    t = jnp.argmax(
        jnp.where(type_legal, jax.random.uniform(k_type, (N_ACTION_TYPES,)), -1.0)
    )
    noise = jax.random.uniform(k_row, (N_FLAT,))
    return jnp.argmax(jnp.where(mask & (t == _ROW_TYPE), noise, -1.0))
