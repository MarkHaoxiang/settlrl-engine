"""Static map from the flat action space to its board structure.

The 662 flat actions are heterogeneous: most are *spatial* (indexed by a vertex,
edge, or tile), a minority are global/resource scalars. A structure-aware policy
head emits each spatial action's logit from the matching board embedding
(``BUILD_ROAD(e)`` is an equivariant per-edge prediction; the robber/knight pick
a tile), not from a flat dense readout. This module precomputes, once, where
every flat action's logit comes from so the head computes compact per-vertex /
-edge / -tile / "other" logits and ``SCATTER``s them into the flat ``N_FLAT``
vector the env masks and targets.

Spatial classes (the head emits one logit per slot per class):

- **vertex** x {setup-settlement, settlement, city}     (from the node embedding)
- **edge**   x {setup-road, road}                        (from endpoint embeddings)
- **tile**   x {robber, knight} x {no-steal, steal}      (from corner-vertex emb)

The *victim* of robber/knight collapses to no-steal vs. steal: our features are
opponent-relative (opponents are not individuated), so a per-victim-player logit
could not be player-relabel invariant -- every "steal a player" action at a tile
shares one logit (legality picks the actual victim). Everything else (resources,
trades, dice, end-turn) is the dense **other** block.

Big-vector layout, gathered by ``SCATTER`` (length ``N_FLAT``, many-to-one on the
tile victims): ``[ vertex | edge | tile | other ]``.
"""

from __future__ import annotations

import numpy as np
from settlrl_agents.internal.rows import ROW_IDX, ROW_TARGET, ROW_TYPE
from settlrl_engine.board.layout import N_EDGES, N_TILES, N_VERTICES
from settlrl_engine.env import N_FLAT, ActionType

_VCLASS = {
    int(ActionType.SETUP_SETTLEMENT): 0,
    int(ActionType.BUILD_SETTLEMENT): 1,
    int(ActionType.BUILD_CITY): 2,
}
_ECLASS = {int(ActionType.SETUP_ROAD): 0, int(ActionType.BUILD_ROAD): 1}
_T_ROBBER, _T_KNIGHT = int(ActionType.MOVE_ROBBER), int(ActionType.PLAY_KNIGHT)
N_VCLASS, N_ECLASS, N_TCLASS = len(_VCLASS), len(_ECLASS), 4  # tile: 2 acts x steal
N_TYPES = len(ActionType)

_RT = np.asarray(ROW_TYPE, np.int64)
_RI = np.asarray(ROW_IDX, np.int64)
_VICTIM = np.asarray(ROW_TARGET, np.int64)
_is_vertex = np.isin(_RT, list(_VCLASS))
_is_edge = np.isin(_RT, list(_ECLASS))
_is_tile = np.isin(_RT, [_T_ROBBER, _T_KNIGHT])
_is_other = ~(_is_vertex | _is_edge | _is_tile)

# tile class: {robber,knight} x {no-steal (victim<0), steal (victim>=0)}.
_tclass = (_RT == _T_KNIGHT).astype(np.int64) * 2 + (_VICTIM >= 0).astype(np.int64)

N_OTHER = int(_is_other.sum())
_V_OFF = 0
_E_OFF = N_VERTICES * N_VCLASS
_T_OFF = _E_OFF + N_EDGES * N_ECLASS
_O_OFF = _T_OFF + N_TILES * N_TCLASS
BIG = _O_OFF + N_OTHER  # width of the concatenated logit vector

# SCATTER[a] = index into the big vector that holds flat action a's logit.
_vc = np.array([_VCLASS.get(int(t), 0) for t in _RT])
_ec = np.array([_ECLASS.get(int(t), 0) for t in _RT])
SCATTER = np.full(N_FLAT, -1, np.int64)
SCATTER[_is_vertex] = _V_OFF + _RI[_is_vertex] * N_VCLASS + _vc[_is_vertex]
SCATTER[_is_edge] = _E_OFF + _RI[_is_edge] * N_ECLASS + _ec[_is_edge]
SCATTER[_is_tile] = _T_OFF + _RI[_is_tile] * N_TCLASS + _tclass[_is_tile]
SCATTER[_is_other] = _O_OFF + np.arange(N_OTHER)
assert (SCATTER >= 0).all() and SCATTER.max() == BIG - 1
assert len(set(SCATTER.tolist())) == BIG  # every big slot is reachable

# TYPE_ID[a] = action type (0..N_TYPES-1): a per-type bias added to every action
# of that type is the class-balance knob (one "how appealing is this type" logit,
# not drowned by the count of spatial slots).
TYPE_ID = _RT.copy()
