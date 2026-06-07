"""The policy interface: a pure function from one game's view to a flat action."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
from jaxtyping import Array, Bool, Int

from catan_engine.env import N_FLAT, Observation

FlatMask = Bool[Array, f"flat={N_FLAT}"]
"""Legality of every concrete flat action for the acting player (one game)."""

FlatAction = Int[Array, ""]
"""A chosen flat action index in ``[0, N_FLAT)``."""


@runtime_checkable
class Policy(Protocol):
    """A single-game decision function, pure and ``jit`` / ``vmap`` compatible.

    ``key`` is a JAX PRNG key, ``obs`` the acting player's partial observation
    (one game, no batch axis), ``mask`` the flat legality of that player's
    moves. Returns the chosen flat action index; decode it with
    :func:`catan_engine.env.flat_to_action`. When ``mask`` has no legal move
    the returned index is arbitrary (the engine rejects it as ``INVALID``).
    """

    def __call__(
        self, key: jax.Array, obs: Observation, mask: FlatMask
    ) -> FlatAction: ...
