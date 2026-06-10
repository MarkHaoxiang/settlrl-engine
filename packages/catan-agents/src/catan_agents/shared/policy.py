"""The seat interfaces: pure decision functions over one game's view.

Two protocols, split by what a seat consumes — neither sees anything the
player wouldn't: :class:`Policy` reads the acting player's partial
observation; :class:`BeliefPolicy` reads the engine's honest
:class:`~catan_engine.belief.BeliefView` (model-based agents rebuild a
concrete world with ``sample_world`` and search there). Both are valid at any
player count; belief sharpness, not the API, varies with the seat count.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol, runtime_checkable

import jax
from jaxtyping import Array, Bool, Int

from catan_engine.belief import BeliefView
from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import IntScalar
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


@runtime_checkable
class BeliefPolicy(Protocol):
    """A single-game decision function over the player's honest world model.

    ``layout`` is one game's board layout, ``view`` everything ``player``
    knows about it (a :class:`~catan_engine.belief.BeliefView`), ``mask`` the
    flat legality of the player's moves. Same return and no-legal-move
    conventions as :class:`Policy`.
    """

    def __call__(
        self,
        key: jax.Array,
        layout: BoardLayout,
        view: BeliefView,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction: ...


@dataclasses.dataclass(frozen=True)
class AgentSpec:
    """A shipped agent: its decision function, input kind, and seat counts.

    ``observes`` says which protocol ``policy`` satisfies (``"observation"`` ->
    :class:`Policy`, ``"belief"`` -> :class:`BeliefPolicy`); ``n_players``
    holds the player counts the agent may be seated at.
    """

    policy: Policy | BeliefPolicy
    observes: Literal["observation", "belief"]
    n_players: frozenset[int]
