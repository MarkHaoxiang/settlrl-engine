"""Unified action dispatch, flat enumeration, and switch-free legality.

The per-action transition cores live in their topical rule modules (``dice``,
``placement``, ``setup``, ``trade``, ``development``, ``robber``, ``turn``); the
shared vocabulary they all use lives in ``common``. This module is the layer on
top of them:

1. A single ``(action_type, params)`` interface over all 15 actions, dispatched
   with ``jax.lax.switch`` (``apply_action`` / ``action_available``) so the whole
   thing stays traceable and vmappable. The heterogeneous per-action params are
   packed into one ``ActionParams`` struct; each branch unpacks what it needs.
2. The *flat action table*: one enumerated index per concrete move (each vertex/
   edge/tile/resource choice plus the parameterless actions; Discard is one
   entry per resource -- one card per action), decoded back to
   ``(ActionType, ActionParams)``. This backs both ``random_actions`` and the
   single-game AEC wrapper's flat ``Discrete`` action space.
3. The switch-free *flat legality* sweep and the per-action-type / per-index
   legality enumerations, which call each core directly over its own static
   slice of the parameter space (no ``lax.switch`` branch blow-up).

All single-game cores are ``jax.vmap``-ed (see ``catan_engine.env``) to run a
whole batch at once. ``ActionResult`` / ``Mask`` / ``ResultCode`` /
``player_total_vp`` are re-exported from ``common`` for callers that import them
from here.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from catan_engine.board.layout import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
    BoardLayout,
)
from catan_engine.board.resources import N_PLAYERS, N_RESOURCES
from catan_engine.board.state import BoardState
from catan_engine.mechanics.awards import resolve_step
from catan_engine.mechanics.common import (
    ActionResult,
    ActionTypeArray,
    IndexAvail,
    IndexParam,
    Mask,
    NoneAvail,
    PairAvail,
    ResultCode,
    player_total_vp,
)
from catan_engine.mechanics.development import (
    _buy_dev_apply,
    _buy_dev_avail,
    _knight_apply,
    _knight_avail,
    _monopoly_apply,
    _monopoly_avail,
    _road_building_apply,
    _road_building_avail,
    _yop_apply,
    _yop_avail,
)
from catan_engine.mechanics.dice import _roll_apply, _roll_avail
from catan_engine.mechanics.placement import (
    _build_city_apply,
    _build_city_avail,
    _build_road_apply,
    _build_road_avail,
    _build_settlement_apply,
    _build_settlement_avail,
)
from catan_engine.mechanics.robber import (
    _discard_apply,
    _discard_avail,
    _move_robber_apply,
    _move_robber_avail,
)
from catan_engine.mechanics.setup import (
    _setup_road_apply,
    _setup_road_avail,
    _setup_settlement_apply,
    _setup_settlement_avail,
)
from catan_engine.mechanics.trade import _maritime_apply, _maritime_avail
from catan_engine.mechanics.turn import _end_turn_apply, _end_turn_avail

__all__ = [
    "ActionParams",
    "ActionResult",
    "ActionType",
    "ActionTypeArray",
    "Mask",
    "N_ACTION_TYPES",
    "ResultCode",
    "action_available",
    "apply_action",
    "flat_legality",
    "player_total_vp",
]


# ===========================================================================
# Unified action dispatch
# ===========================================================================


class ActionType(IntEnum):
    """Index into the unified action set (the order of the dispatch branches)."""

    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL_DICE = 2
    DISCARD = 3
    MOVE_ROBBER = 4
    BUILD_ROAD = 5
    BUILD_SETTLEMENT = 6
    BUILD_CITY = 7
    BUY_DEVELOPMENT_CARD = 8
    PLAY_KNIGHT = 9
    PLAY_ROAD_BUILDING = 10
    PLAY_YEAR_OF_PLENTY = 11
    PLAY_MONOPOLY = 12
    MARITIME_TRADE = 13
    END_TURN = 14


N_ACTION_TYPES = len(ActionType)


class ActionParams(NamedTuple):
    """Packed parameters for the unified ``apply_action`` / ``action_available``.

    Single game: all fields are 0-d arrays (batch by vmapping). Each action
    reads only what it needs; unused fields are ignored. By action:

    - vertex/edge/tile/resource/give (single index)  -> ``idx``
      (Discard's ``idx`` is the resource to give up one card of)
    - victim / receive / second resource (Year of Plenty) -> ``target``
    - parameterless actions (RollDice, BuyDevelopmentCard, PlayRoadBuilding,
      EndTurn) read nothing.

    ``target`` follows the ``victim == -1`` ("steal from no one") convention for
    the robber actions. (``idx`` rather than ``index`` because ``NamedTuple``
    reserves the ``index`` attribute.)
    """

    idx: IndexParam  # primary index
    target: IndexParam  # secondary index (victim / receive / resource_b)


# Branch adapters in ActionType order: each maps the packed ActionParams onto the
# single-game core's native param shape. Each core takes the precomputed
# ``available`` legality (the caller computes it once -- see ``apply_action``); no
# branch recomputes avail.
_APPLY_BRANCHES = (
    lambda lay, st, pp, av: _setup_settlement_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _setup_road_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _roll_apply(lay, st, None, av),
    lambda lay, st, pp, av: _discard_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _move_robber_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _build_road_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _build_settlement_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _build_city_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _buy_dev_apply(lay, st, None, av),
    lambda lay, st, pp, av: _knight_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _road_building_apply(lay, st, None, av),
    lambda lay, st, pp, av: _yop_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _monopoly_apply(lay, st, pp.idx, av),
    lambda lay, st, pp, av: _maritime_apply(lay, st, (pp.idx, pp.target), av),
    lambda lay, st, pp, av: _end_turn_apply(lay, st, None, av),
)

_AVAIL_BRANCHES = (
    lambda lay, st, pp: _setup_settlement_avail(lay, st, pp.idx),
    lambda lay, st, pp: _setup_road_avail(lay, st, pp.idx),
    lambda lay, st, pp: _roll_avail(lay, st, None),
    lambda lay, st, pp: _discard_avail(lay, st, pp.idx),
    lambda lay, st, pp: _move_robber_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _build_road_avail(lay, st, pp.idx),
    lambda lay, st, pp: _build_settlement_avail(lay, st, pp.idx),
    lambda lay, st, pp: _build_city_avail(lay, st, pp.idx),
    lambda lay, st, pp: _buy_dev_avail(lay, st, None),
    lambda lay, st, pp: _knight_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _road_building_avail(lay, st, None),
    lambda lay, st, pp: _yop_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _monopoly_avail(lay, st, pp.idx),
    lambda lay, st, pp: _maritime_avail(lay, st, (pp.idx, pp.target)),
    lambda lay, st, pp: _end_turn_avail(lay, st, None),
)


def apply_action(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
    available: Mask,
) -> tuple[BoardState, ResultCode]:
    """Apply ``action_type`` (single game) and return (new state, ActionResult code).

    ``available`` is the precomputed legality of this ``(action_type, params)``
    move: the caller computes avail *once* and passes it in, and the chosen branch
    gates its core state change on it (``tree_select`` / INVALID). No branch
    recomputes avail -- under ``vmap`` every branch runs regardless of which action
    was chosen, so an internal avail call would be paid ~15x per lane. Callers
    obtain ``available`` either from :func:`action_available` (the switch-based
    avail, exact for any params) or from a cached flat-legality sweep (see
    :func:`flat_legality`).

    Two stages: the ``lax.switch`` applies the chosen action's *core* state change
    (stage 1), then :func:`awards.resolve_step` recomputes the awards and resolves
    the win *once* (stage 2). Keeping the award sweep out of the per-action
    branches avoids running the expensive Longest Road DFS in every branch.
    """
    state, result = jax.lax.switch(
        action_type, _APPLY_BRANCHES, layout, state, params, available
    )
    return resolve_step(state, result)


def action_available(
    layout: BoardLayout,
    state: BoardState,
    action_type: ActionTypeArray,
    params: ActionParams,
) -> Mask:
    """Legality of ``action_type`` with ``params`` (single game) as a scalar bool."""
    return cast(
        Mask,
        jax.lax.switch(action_type, _AVAIL_BRANCHES, layout, state, params),
    )


# ===========================================================================
# Flat action table: one enumerated index per concrete move.
#
# A single flat enumeration of every concrete action (each vertex/edge/tile/
# resource choice plus the parameterless moves), decoded to the engine's
# ``(ActionType, ActionParams)``. DISCARD is per-resource (one card of that
# resource per action), so the table is fully static -- no per-lane fill-in.
# This is the table behind both :meth:`BatchedCatanEnv.random_actions` and the
# single-game AEC wrapper's flat ``Discrete`` action space (see ``env/aec.py``).
# ===========================================================================
def _build_action_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat action index -> (ActionType, primary index, secondary target)."""
    atype: list[int] = []
    idx: list[int] = []
    target: list[int] = []

    def add(t: ActionType, i: int = 0, tg: int = 0) -> None:
        atype.append(int(t))
        idx.append(i)
        target.append(tg)

    for v in range(N_VERTICES):
        add(ActionType.SETUP_SETTLEMENT, v)
    for e in range(N_EDGES):
        add(ActionType.SETUP_ROAD, e)
    add(ActionType.ROLL_DICE)
    for r in range(N_RESOURCES):
        add(ActionType.DISCARD, r)
    for t in range(N_TILES):
        for victim in range(-1, N_PLAYERS):
            add(ActionType.MOVE_ROBBER, t, victim)
    for e in range(N_EDGES):
        add(ActionType.BUILD_ROAD, e)
    for v in range(N_VERTICES):
        add(ActionType.BUILD_SETTLEMENT, v)
    for v in range(N_VERTICES):
        add(ActionType.BUILD_CITY, v)
    add(ActionType.BUY_DEVELOPMENT_CARD)
    for t in range(N_TILES):
        for victim in range(-1, N_PLAYERS):
            add(ActionType.PLAY_KNIGHT, t, victim)
    add(ActionType.PLAY_ROAD_BUILDING)
    for a in range(N_RESOURCES):
        for b in range(N_RESOURCES):
            add(ActionType.PLAY_YEAR_OF_PLENTY, a, b)
    for r in range(N_RESOURCES):
        add(ActionType.PLAY_MONOPOLY, r)
    for g in range(N_RESOURCES):
        for r in range(N_RESOURCES):
            add(ActionType.MARITIME_TRADE, g, r)
    add(ActionType.END_TURN)

    return (
        np.asarray(atype, dtype=np.int32),
        np.asarray(idx, dtype=np.int32),
        np.asarray(target, dtype=np.int32),
    )


_ATYPE, _IDX, _TARGET = _build_action_table()
_N_FLAT = int(_ATYPE.shape[0])
# Device-side copies for the batched random-action sweep.
_ATYPE_J = jnp.asarray(_ATYPE)
_IDX_J = jnp.asarray(_IDX)
_TARGET_J = jnp.asarray(_TARGET)


# ===========================================================================
# Reverse lookup: (action_type, idx, target) -> flat row.
#
# Lets a caller read one ``(action_type, params)`` action's legality straight out
# of a cached ``(B, N_FLAT)`` flat-legality sweep (:func:`flat_legality`) instead
# of recomputing avail for the chosen action. The flat table holds one row per
# concrete move, so ``(action_type, idx, target)`` is unique per row; we pack the
# triple into a dense int32 index and store its row.
# ===========================================================================
_IDX_RANGE = int(_IDX.max()) + 1
_TGT_MIN = int(_TARGET.min())
_TGT_RANGE = int(_TARGET.max()) - _TGT_MIN + 1
_REVERSE = np.zeros(N_ACTION_TYPES * _IDX_RANGE * _TGT_RANGE, dtype=np.int32)
_REVERSE[(_ATYPE * _IDX_RANGE + _IDX) * _TGT_RANGE + (_TARGET - _TGT_MIN)] = np.arange(
    _N_FLAT, dtype=np.int32
)
_REVERSE_J = jnp.asarray(_REVERSE)


def flat_legality(
    avail_flat: jax.Array,
    action_type: ActionTypeArray,
    idx: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """``(B,)`` legality of each lane's ``(action_type, idx, target)`` move.

    Maps the action back to its flat row and gathers the bit from the cached
    ``(B, N_FLAT)`` mask ``avail_flat``. An action that is not a row of the flat
    table (out-of-range params) reads ``False`` (the row decode fails the identity
    check).
    """
    pack = (action_type * _IDX_RANGE + idx) * _TGT_RANGE + (target - _TGT_MIN)
    pack = jnp.clip(pack, 0, _REVERSE_J.shape[0] - 1)
    row = _REVERSE_J[pack]  # (B,)
    matches = (
        (_ATYPE_J[row] == action_type)
        & (_IDX_J[row] == idx)
        & (_TARGET_J[row] == target)
    )
    rows = jnp.arange(avail_flat.shape[0])
    return avail_flat[rows, row] & matches


# ===========================================================================
# Switch-free flat legality sweep.
#
# ``action_available`` dispatches the 15 action types with ``jax.lax.switch``.
# ``vmap``-ing it over the whole flat table (whose action type *varies* per
# entry) forces XLA to evaluate *every* branch for *every* entry -- a ~15x
# blow-up, worse once the expensive cores (Longest Road, placement scatters)
# are counted. Because the flat table is *static* we instead call each action's
# legality core *directly* over its own slice of the table, board closed over,
# and stitch the results back in table order: same mask, no switch.
# ===========================================================================
def _flat_positions(t: ActionType) -> np.ndarray:
    """Flat-table row indices belonging to action type ``t`` (static)."""
    return np.where(_ATYPE == int(t))[0]


# (core, flat rows, primary index per row) for the single-index actions.
_INDEX_AVAIL: tuple[tuple[IndexAvail, jax.Array, jax.Array], ...] = tuple(
    (core, jnp.asarray(p), jnp.asarray(_IDX[p]))
    for core, p in (
        (_setup_settlement_avail, _flat_positions(ActionType.SETUP_SETTLEMENT)),
        (_setup_road_avail, _flat_positions(ActionType.SETUP_ROAD)),
        (_discard_avail, _flat_positions(ActionType.DISCARD)),
        (_build_road_avail, _flat_positions(ActionType.BUILD_ROAD)),
        (_build_settlement_avail, _flat_positions(ActionType.BUILD_SETTLEMENT)),
        (_build_city_avail, _flat_positions(ActionType.BUILD_CITY)),
        (_monopoly_avail, _flat_positions(ActionType.PLAY_MONOPOLY)),
    )
)

# (core, flat rows, primary, secondary) for the (idx, target) pair actions.
_PAIR_AVAIL: tuple[tuple[PairAvail, jax.Array, jax.Array, jax.Array], ...] = tuple(
    (core, jnp.asarray(p), jnp.asarray(_IDX[p]), jnp.asarray(_TARGET[p]))
    for core, p in (
        (_move_robber_avail, _flat_positions(ActionType.MOVE_ROBBER)),
        (_knight_avail, _flat_positions(ActionType.PLAY_KNIGHT)),
        (_yop_avail, _flat_positions(ActionType.PLAY_YEAR_OF_PLENTY)),
        (_maritime_avail, _flat_positions(ActionType.MARITIME_TRADE)),
    )
)

# (core, single flat row) for the parameterless actions.
_NONE_AVAIL: tuple[tuple[NoneAvail, int], ...] = tuple(
    (core, int(_flat_positions(t)[0]))
    for core, t in (
        (_roll_avail, ActionType.ROLL_DICE),
        (_buy_dev_avail, ActionType.BUY_DEVELOPMENT_CARD),
        (_road_building_avail, ActionType.PLAY_ROAD_BUILDING),
        (_end_turn_avail, ActionType.END_TURN),
    )
)


def _sweep_index(
    core: IndexAvail, layout: BoardLayout, state: BoardState, idxs: jax.Array
) -> jax.Array:
    """Map a single-index legality core over ``idxs`` (board closed over)."""
    return jax.vmap(lambda i: core(layout, state, i))(idxs)


def _sweep_pair(
    core: PairAvail,
    layout: BoardLayout,
    state: BoardState,
    idxs: jax.Array,
    tgts: jax.Array,
) -> jax.Array:
    """Map a ``(primary, secondary)`` legality core over ``idxs``/``tgts``."""
    return jax.vmap(lambda i, tg: core(layout, state, (i, tg)))(idxs, tgts)


def _flat_available_for(layout: BoardLayout, state: BoardState) -> jax.Array:
    """``(N_FLAT,)`` legality of every flat action for one game -- switch-free.

    Each action type's legality core is ``vmap``-ed over its own slice of the
    flat table (board closed over, not replicated) and scattered back into the
    flat mask, reproducing :func:`action_available` over the table without the
    ``lax.switch`` branch blow-up.
    """
    out = jnp.zeros(_N_FLAT, dtype=bool)
    for core, pos, idxs in _INDEX_AVAIL:
        out = out.at[pos].set(_sweep_index(core, layout, state, idxs))
    for pair_core, pos, idxs, tgts in _PAIR_AVAIL:
        out = out.at[pos].set(_sweep_pair(pair_core, layout, state, idxs, tgts))
    for none_core, p in _NONE_AVAIL:
        out = out.at[p].set(none_core(layout, state, None))
    return out


_flat_available_b = jax.jit(jax.vmap(_flat_available_for))
"""``(B, N_FLAT)`` flat legality per lane for its acting player (switch-free)."""


# ===========================================================================
# Per-index legality enumerations.
#
# A finer view over the same cores than the flat sweep: ``_INDEX_MASKS`` sweeps an
# index-parameterized action's whole primary domain (the env's
# ``available_indices``). The per-action-type "is any move of this type legal"
# mask (the env's ``action_mask``) is no longer a separate sweep -- the env
# reduces it straight from the cached flat-legality mask.
# ===========================================================================

# Static parameter domains for the robber-tile legality sweep.
_TILE_DOM = jnp.arange(N_TILES, dtype=jnp.int32)
_VICTIM_DOM = jnp.arange(-1, N_PLAYERS, dtype=jnp.int32)  # -1 = steal from no one


_BatchedMask = Callable[[BoardLayout, BoardState], jax.Array]


def _index_mask_factory(avail: IndexAvail, n: int) -> _BatchedMask:
    """Batched ``(B, n)`` legality sweep over a single index parameter."""
    dom = jnp.arange(n, dtype=jnp.int32)

    def single(layout: BoardLayout, state: BoardState) -> jax.Array:
        return jax.vmap(lambda i: avail(layout, state, i))(dom)

    return cast(_BatchedMask, jax.jit(jax.vmap(single)))


def _robber_tile_mask_factory(avail: PairAvail) -> _BatchedMask:
    """Batched ``(B, N_TILES)`` mask: a tile is legal if some victim choice works."""

    def single(layout: BoardLayout, state: BoardState) -> jax.Array:
        return jax.vmap(
            lambda t: jnp.any(
                jax.vmap(lambda v: avail(layout, state, (t, v)))(_VICTIM_DOM)
            )
        )(_TILE_DOM)

    return cast(_BatchedMask, jax.jit(jax.vmap(single)))


# ActionType -> batched legality sweep over that action's primary index domain.
# (Multi-parameter / parameterless actions are absent; use ``action_mask`` /
# ``available`` for those.)
_INDEX_MASKS = {
    ActionType.SETUP_SETTLEMENT: _index_mask_factory(
        _setup_settlement_avail, N_VERTICES
    ),
    ActionType.SETUP_ROAD: _index_mask_factory(_setup_road_avail, N_EDGES),
    ActionType.DISCARD: _index_mask_factory(_discard_avail, N_RESOURCES),
    ActionType.BUILD_ROAD: _index_mask_factory(_build_road_avail, N_EDGES),
    ActionType.BUILD_SETTLEMENT: _index_mask_factory(
        _build_settlement_avail, N_VERTICES
    ),
    ActionType.BUILD_CITY: _index_mask_factory(_build_city_avail, N_VERTICES),
    ActionType.PLAY_MONOPOLY: _index_mask_factory(_monopoly_avail, N_RESOURCES),
    ActionType.MOVE_ROBBER: _robber_tile_mask_factory(_move_robber_avail),
    ActionType.PLAY_KNIGHT: _robber_tile_mask_factory(_knight_avail),
}
