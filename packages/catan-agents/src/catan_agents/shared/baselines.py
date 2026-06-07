"""Baseline policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.env import N_FLAT, Observation

from catan_agents.shared.policy import FlatAction, FlatMask


def random_policy(key: jax.Array, obs: Observation, mask: FlatMask) -> FlatAction:
    """Uniform over the legal flat actions (``obs`` is ignored)."""
    noise = jax.random.uniform(key, (N_FLAT,))
    return jnp.argmax(jnp.where(mask, noise, -1.0))
