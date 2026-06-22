"""The search's value seam: how good is this board for a given player?

A :class:`ValueFunction` scores a concrete board. The search agents only ever
hand it *sampled* worlds (see :mod:`settlrl_search.sample`), so the "hidden"
fields it reads there are belief-consistent samples.

The shipped heuristic and the classical-fit deployments live above this seam,
in ``settlrl_agents.value``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import BoardState, Player

Value = Float[Array, ""]
"""A scalar state score for one player: higher is better, arbitrary scale."""


@runtime_checkable
class ValueFunction(Protocol):
    """A single-game state evaluation, pure and ``jit`` / ``vmap`` compatible.

    ``layout`` / ``state`` are one game's board (no batch axis); returns the
    state's value from ``player``'s point of view.
    """

    def __call__(
        self, layout: BoardLayout, state: BoardState, player: Player
    ) -> Value: ...
