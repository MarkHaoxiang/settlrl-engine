"""Shared type aliases for the ISMCTS tree modules."""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, Float, Int
from settlrl_engine.belief import BeliefView
from settlrl_engine.board.layout import BoardLayout
from settlrl_engine.board.state import IntScalar, KeyScalar, Player
from settlrl_engine.env import N_FLAT

from settlrl_search._common import _Weights
from settlrl_search.value import Value

_FlatVec = Float[Array, f"flat={N_FLAT}"]  # a value per flat action
_LegalMask = _FlatVec  # 1.0 on the legal flat actions, else 0.0
_PriorLogits = _FlatVec  # prior logits over the flat actions
_NodeI = Int[Array, "node"]
_NodeF = Float[Array, "node"]
_EdgeI = Int[Array, "node act"]
_Table = Int[Array, "m sims"]
_EdgeF = Float[Array, "node act"]
_PathI = Int[Array, "depth"]

# Search-local scalars (the engine's IntScalar, named for their role).
_Node = IntScalar  # a node id in the _Tree (0 = root)
_Action = IntScalar  # a flat action index

TreeSearch = Callable[
    [KeyScalar, BoardLayout, BeliefView, Player, _LegalMask, _PriorLogits],
    tuple[_Weights, Value],
]
"""One ISMCTS search (what :func:`make_tree` returns): the searcher's legal set
and root prior in, the improved-policy ``action_weights`` and the searched root
value (searcher frame) out."""
