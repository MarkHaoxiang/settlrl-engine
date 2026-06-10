"""Decode the AEC flat action set into JSON-friendly action descriptors.

The engine enumerates every concrete move as a flat ``Discrete`` index
(``catan_engine.env.N_FLAT`` rows, decoded with ``flat_to_action``). The
renderer needs each legal index turned into something the frontend can act on:
the action type, a human label, and — for placement / robber / resource moves —
the board geometry or resources involved, expressed in the same cube/axial
coordinates the SVG board already uses (reused from
:mod:`catan_render.convert`).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from catan_engine.env import N_FLAT, ActionType, flat_to_action

from .convert import _RESOURCE_NAMES, EDGE_VERTICES, TILE_COORDS, VERTEX_COORDS, _cube
from .models import ActionModel, EdgeModel, HexModel

__all__ = ["_RESOURCE_NAMES", "_decode", "decode_actions"]

# Host-side copy of the engine's flat action table: row -> (type, idx, target).
_row_type, _row_params = flat_to_action(jnp.arange(N_FLAT))
_ATYPE = np.asarray(_row_type)
_IDX = np.asarray(_row_params.idx)
_TARGET = np.asarray(_row_params.target)

# Action types grouped by the kind of board target they carry.
_VERTEX_TYPES = {
    ActionType.SETUP_SETTLEMENT,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
}
_ROAD_TYPES = {ActionType.SETUP_ROAD, ActionType.BUILD_ROAD}
_ROBBER_TYPES = {ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT}

_BASE_LABELS = {
    ActionType.SETUP_SETTLEMENT: "Settlement",
    ActionType.BUILD_SETTLEMENT: "Settlement",
    ActionType.BUILD_CITY: "City",
    ActionType.SETUP_ROAD: "Road",
    ActionType.BUILD_ROAD: "Road",
    ActionType.ROLL_DICE: "Roll dice",
    ActionType.END_TURN: "End turn",
    ActionType.BUY_DEVELOPMENT_CARD: "Buy dev card",
    ActionType.PLAY_ROAD_BUILDING: "Road building",
}


def _decode(flat: int) -> ActionModel:
    """Turn one flat action index into an :class:`ActionModel`."""
    at = ActionType(int(_ATYPE[flat]))
    idx = int(_IDX[flat])
    target = int(_TARGET[flat])
    type_name = at.name.lower()

    if at in _VERTEX_TYPES:
        label = "City" if at is ActionType.BUILD_CITY else "Settlement"
        return ActionModel(
            flat=flat, type=type_name, label=label, vertex=_cube(VERTEX_COORDS[idx])
        )

    if at in _ROAD_TYPES:
        v1, v2 = EDGE_VERTICES[idx]
        edge = EdgeModel(a=_cube(VERTEX_COORDS[v1]), b=_cube(VERTEX_COORDS[v2]))
        return ActionModel(flat=flat, type=type_name, label="Road", edge=edge)

    if at in _ROBBER_TYPES:
        q, r = TILE_COORDS[idx]
        verb = "Knight" if at is ActionType.PLAY_KNIGHT else "Move robber"
        steal = f" (steal P{target + 1})" if target >= 0 else ""
        return ActionModel(
            flat=flat,
            type=type_name,
            label=f"{verb}{steal}",
            tile=HexModel(q=q, r=r),
            victim=target,
        )

    if at is ActionType.DISCARD:
        res = _RESOURCE_NAMES[idx]
        return ActionModel(
            flat=flat, type=type_name, label=f"Discard {res}", resource=res
        )

    if at is ActionType.PLAY_MONOPOLY:
        res = _RESOURCE_NAMES[idx]
        return ActionModel(
            flat=flat, type=type_name, label=f"Monopoly: {res}", resource=res
        )

    if at is ActionType.PLAY_YEAR_OF_PLENTY:
        a, b = _RESOURCE_NAMES[idx], _RESOURCE_NAMES[target]
        return ActionModel(
            flat=flat, type=type_name, label=f"Plenty: {a} + {b}", resources=[a, b]
        )

    if at is ActionType.MARITIME_TRADE:
        give, receive = _RESOURCE_NAMES[idx], _RESOURCE_NAMES[target]
        return ActionModel(
            flat=flat,
            type=type_name,
            label=f"Trade {give} → {receive}",
            give=give,
            receive=receive,
        )

    return ActionModel(flat=flat, type=type_name, label=_BASE_LABELS.get(at, type_name))


def decode_actions(flat_indices: list[int]) -> list[ActionModel]:
    """Decode a list of legal flat action indices into action descriptors."""
    return [_decode(f) for f in flat_indices]
