"""Board symmetries and player relabelings, for the architecture invariance tests.

In cube coordinates every hexagon symmetry is a *signed permutation of the three
axes*: the six axis permutations are the rotations/reflections that fix the plane
``q+r+s=const`` and the sign flip is the central (180-degree) point symmetry --
the dihedral group D6 (order 12) of the vertex/edge/tile graph. The standard
*harbor* placement is only 3-fold symmetric, so the symmetries of the full board
(graph plus ports) are the order-6 subgroup (D3: 3 rotations + 3 reflections)
that map port slots to port slots. ``board_symmetries`` returns that subgroup;
each element is a graph automorphism, so applying one produces a genuinely
rotated/reflected position a structure-aware net must score identically.

``relabel_players`` is the orthogonal symmetry: swap the player identities (and
the perspective) and a player-relative featurization is unchanged.
"""

from __future__ import annotations

from itertools import permutations
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from settlrl_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    PORT_V,
    BoardLayout,
    edge_cubes,
    edge_index,
    tile_cube,
    tile_index,
    vertex_cube,
    vertex_index,
)
from settlrl_engine.board.state import BoardState
from settlrl_engine.env import N_FLAT, ActionType
from settlrl_search.rows import ROW_IDX, ROW_TARGET, ROW_TYPE, flat_row

Cube = tuple[int, int, int]
_AxisOp = tuple[tuple[int, ...], int]  # (axis permutation, sign)


def _apply(op: _AxisOp, cube: Cube) -> Cube:
    axes, sign = op
    return (sign * cube[axes[0]], sign * cube[axes[1]], sign * cube[axes[2]])


class Symmetry(NamedTuple):
    """One board automorphism as the new index of every vertex / edge / tile /
    port (each a permutation of ``range(n)``)."""

    vertices: np.ndarray
    edges: np.ndarray
    tiles: np.ndarray
    ports: np.ndarray


def board_symmetries() -> list[Symmetry]:
    """The order-6 symmetry group of the full board (graph + harbors): the signed
    axis permutations whose vertex map also sends every port slot to a port slot.
    The identity is included; each component is asserted a permutation."""
    port_of = {
        frozenset(int(v) for v in vs): p for p, vs in enumerate(np.asarray(PORT_V))
    }
    syms: list[Symmetry] = []
    for axes in permutations((0, 1, 2)):
        for sign in (1, -1):
            op: _AxisOp = (axes, sign)
            vert = np.array(
                [vertex_index(_apply(op, vertex_cube(v))) for v in range(N_VERTICES)]
            )
            images = [frozenset(int(vert[v]) for v in vs) for vs in np.asarray(PORT_V)]
            if any(img not in port_of for img in images):
                continue  # this op rotates a port slot off the harbors -- not a symmetry
            tiles = np.array(
                [tile_index(_apply(op, tile_cube(t))) for t in range(N_TILES)]
            )
            edges = np.array(
                [
                    edge_index(_apply(op, a), _apply(op, b))
                    for a, b in (edge_cubes(e) for e in range(N_EDGES))
                ]
            )
            ports = np.array([port_of[img] for img in images])
            for perm, n in (
                (vert, N_VERTICES),
                (edges, N_EDGES),
                (tiles, N_TILES),
                (ports, N_PORTS),
            ):
                assert sorted(perm.tolist()) == list(range(n)), "not a permutation"
            syms.append(Symmetry(vert, edges, tiles, ports))
    return syms


def _inv(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0])
    return inv


def apply_symmetry(
    layout: BoardLayout, state: BoardState, sym: Symmetry
) -> tuple[BoardLayout, BoardState]:
    """Rotate/reflect a single-game board by ``sym``. Geometry-indexed arrays are
    gathered to their new positions; the robber's tile index is value-remapped.
    Players are untouched."""
    vi, ei, ti = _inv(sym.vertices), _inv(sym.edges), _inv(sym.tiles)
    pi = _inv(sym.ports)
    layout2 = layout._replace(
        tile_resource=layout.tile_resource[ti],
        tile_number=layout.tile_number[ti],
        port_allocation=layout.port_allocation[pi],
    )
    state2 = state._replace(
        vertex_owner=state.vertex_owner[vi],
        vertex_type=state.vertex_type[vi],
        edge_road=state.edge_road[ei],
        robber=jnp.asarray(sym.tiles)[state.robber].astype(jnp.uint8),
    )
    return layout2, state2


def relabel_players(state: BoardState, perm: np.ndarray) -> BoardState:
    """Relabel player identities: old player ``i`` becomes ``perm[i]``. Per-player
    rows are reordered; occupancy/award/turn values are remapped (the NO_INDEX
    sentinel survives, being out of player range)."""
    P = state.n_players
    pj = jnp.asarray(perm)
    inv = jnp.asarray(_inv(perm))

    def remap_owner(arr: jnp.ndarray) -> jnp.ndarray:  # 0 = empty, else player + 1
        v = arr.astype(jnp.int32)
        return jnp.where(v > 0, pj[jnp.clip(v - 1, 0, P - 1)] + 1, 0).astype(jnp.uint8)

    def remap_idx(arr: jnp.ndarray) -> jnp.ndarray:  # player index or NO_INDEX
        v = arr.astype(jnp.int32)
        return jnp.where(v < P, pj[jnp.clip(v, 0, P - 1)], v).astype(jnp.uint8)

    return state._replace(
        vertex_owner=remap_owner(state.vertex_owner),
        edge_road=remap_owner(state.edge_road),
        player_resources=state.player_resources[inv],
        victory_points=state.victory_points[inv],
        dev_hand=state.dev_hand[inv],
        knights_played=state.knights_played[inv],
        pending_discard=state.pending_discard[inv],
        current_player=remap_idx(state.current_player),
        trade_partner=remap_idx(state.trade_partner),
        longest_road_owner=remap_idx(state.longest_road_owner),
        largest_army_owner=remap_idx(state.largest_army_owner),
    )


def action_permutation(sym: Symmetry) -> np.ndarray:
    """The flat-action permutation induced by board symmetry ``sym``: an action's
    image is the same action type at the symmetry-mapped vertex / edge / tile
    (non-spatial actions map to themselves). For the policy *equivariance* test:
    ``policy(apply_symmetry(board))[action_permutation(sym)] == policy(board)``."""
    rt, ri, tg = np.asarray(ROW_TYPE), np.asarray(ROW_IDX), np.asarray(ROW_TARGET)
    vt = {int(a) for a in (ActionType.SETUP_SETTLEMENT, ActionType.BUILD_SETTLEMENT,
                           ActionType.BUILD_CITY)}  # fmt: skip
    et = {int(ActionType.SETUP_ROAD), int(ActionType.BUILD_ROAD)}
    tt = {int(ActionType.MOVE_ROBBER), int(ActionType.PLAY_KNIGHT)}
    perm = np.arange(N_FLAT)
    for a in range(N_FLAT):
        t = int(rt[a])
        slot = (
            sym.vertices if t in vt else sym.edges if t in et else sym.tiles
            if t in tt else None
        )  # fmt: skip
        if slot is not None:
            perm[a] = flat_row(ActionType(t), int(slot[ri[a]]), int(tg[a]))
    assert sorted(perm.tolist()) == list(range(N_FLAT)), "not a permutation"
    return perm
