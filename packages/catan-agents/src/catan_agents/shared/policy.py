"""The seat interfaces: pure decision functions over one game's view.

Two protocols, split by what a seat may legitimately see:

- :class:`Policy` consumes the acting player's *partial observation* and is
  valid at any player count.
- :class:`StatePolicy` consumes the full single-game board. With two players
  every resource flow is publicly inferable (production follows from dice and
  board, build/trade costs are public, every steal involves you), so reading
  the engine state is bookkeeping, not cheating; dev cards are the one hidden
  element and are only a distribution over the known deck composition. With
  3-4 players opponent-to-opponent steals and discards are hidden, so state
  seats are restricted to two-player games.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol, runtime_checkable

import jax
from jaxtyping import Array, Bool, Int

from catan_engine.board.layout import BoardLayout
from catan_engine.board.state import BoardState, IntScalar
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
class StatePolicy(Protocol):
    """A single-game decision function over the full board state.

    ``layout`` / ``state`` are one game's board (no batch axis), ``player`` the
    seat deciding, ``mask`` the flat legality of that player's moves. Same
    return and no-legal-move conventions as :class:`Policy`.
    """

    def __call__(
        self,
        key: jax.Array,
        layout: BoardLayout,
        state: BoardState,
        player: IntScalar,
        mask: FlatMask,
    ) -> FlatAction: ...


@dataclasses.dataclass(frozen=True)
class AgentSpec:
    """A shipped agent: its decision function, input kind, and seat counts.

    ``observes`` says which protocol ``policy`` satisfies (``"observation"`` ->
    :class:`Policy`, ``"state"`` -> :class:`StatePolicy`); ``n_players`` holds
    the player counts the agent may be seated at.
    """

    policy: Policy | StatePolicy
    observes: Literal["observation", "state"]
    n_players: frozenset[int]
