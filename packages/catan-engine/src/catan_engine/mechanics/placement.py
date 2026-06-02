"""Settlement / road placement: legality geometry and the build action cores.

Vertex incidence is computed by scattering per-edge values onto their endpoints
over the COO ``edge_index`` (``EDGE_V``), rather than gathering through padded
vertex->edge reverse maps -- the JAX analogue of PyG message passing.

The lower half holds the ``BuildRoad`` / ``BuildSettlement`` / ``BuildCity``
action cores, which compose the placement-legality predicates above with the
economy helpers in ``common``. The cores apply only the core state change; the
Longest Road award and win check are resolved once per step by
``awards.resolve_step`` (run after dispatch, not inside each core).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from catan_engine.board import Board
from catan_engine.board.layout import EDGE_V, N_EDGES, N_VERTICES, BoardLayout
from catan_engine.board.state import (
    CITY,
    MAX_CITIES,
    MAX_SETTLEMENTS,
    SETTLEMENT,
    BoardState,
    BoolScalar,
    EdgeRoadVec,
    IntScalar,
    VertexOwnerVec,
    to_u8,
    tree_select,
)
from catan_engine.mechanics import awards
from catan_engine.mechanics.common import (
    CITY_COST_ARR,
    INVALID,
    ROAD_COST_ARR,
    SETTLEMENT_COST_ARR,
    SUCCESS,
    IndexParam,
    Mask,
    ResultCode,
    can_afford,
    count_cities,
    count_settlements,
    main_after_roll,
    pay,
    roads_left,
)

# COO edge_index endpoints (static graph connectivity).
_SRC = EDGE_V[:, 0]
_DST = EDGE_V[:, 1]


def _scatter_to_vertices(per_edge: jax.Array) -> jax.Array:
    """Sum a per-edge value onto both of each edge's endpoints."""
    acc = jnp.zeros((N_VERTICES,), per_edge.dtype)
    return acc.at[_SRC].add(per_edge).at[_DST].add(per_edge)


def distance_rule_ok(vertex_owner: VertexOwnerVec, vertex: IntScalar) -> BoolScalar:
    """Vertex empty and no adjacent vertex carries a building."""
    occ = (vertex_owner != 0).astype(jnp.int32)  # (N_VERTICES,)
    # Scatter each edge's endpoint occupancy onto the opposite endpoint.
    nbr_occ = jnp.zeros_like(occ).at[_SRC].add(occ[_DST]).at[_DST].add(occ[_SRC])
    return (vertex_owner[vertex] == 0) & (nbr_occ[vertex] == 0)


def settlement_connected(
    edge_road: EdgeRoadVec, player: IntScalar, vertex: IntScalar
) -> BoolScalar:
    """Player owns a road incident to ``vertex`` (required outside setup)."""
    mine = (edge_road == player + 1).astype(jnp.int32)  # (N_EDGES,)
    inc = _scatter_to_vertices(mine)  # roads-of-mine touching each vertex
    return inc[vertex] > 0


def setup_road_placeable(
    edge_road: EdgeRoadVec,
    vertex_owner: VertexOwnerVec,
    player: IntScalar,
    edge: IntScalar,
) -> BoolScalar:
    """Setup road: edge touches a player-owned vertex with no incident own road."""
    target = player + 1
    inc = _scatter_to_vertices((edge_road == target).astype(jnp.int32))

    def end_ok(v: jax.Array) -> jax.Array:
        return (vertex_owner[v] == target) & (inc[v] == 0)

    return end_ok(EDGE_V[edge, 0]) | end_ok(EDGE_V[edge, 1])


def road_placeable(
    edge_road: EdgeRoadVec,
    vertex_owner: VertexOwnerVec,
    player: IntScalar,
    edge: IntScalar,
) -> BoolScalar:
    """Edge empty and connects to the player's network at a non-blocked end."""
    target = player + 1
    empty = edge_road[edge] == 0

    own_here = vertex_owner == target  # (N_VERTICES,)
    blocked = (vertex_owner != 0) & ~own_here  # opponent building blocks
    # Own roads incident to each vertex. The candidate edge is empty (gated by
    # ``empty``), so it is never one of mine and never self-counts.
    inc = _scatter_to_vertices((edge_road == target).astype(jnp.int32))

    def end_ok(v: jax.Array) -> jax.Array:
        return own_here[v] | (~blocked[v] & (inc[v] > 0))

    return empty & (end_ok(EDGE_V[edge, 0]) | end_ok(EDGE_V[edge, 1]))


# ===========================================================================
# BuildRoad
# ===========================================================================


def _build_road_avail(
    layout: BoardLayout, state: BoardState, edge: IntScalar
) -> BoolScalar:
    in_range = (edge >= 0) & (edge < N_EDGES)
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    main = main_after_roll(state)
    has_road = roads_left(state.edge_road, player) > 0
    placeable = road_placeable(state.edge_road, state.vertex_owner, player, e)
    free = state.free_roads > 0
    afford = can_afford(state.player_resources[player], ROAD_COST_ARR)
    return in_range & main & has_road & placeable & (free | afford)


def _build_road_apply(
    layout: BoardLayout, state: BoardState, edge: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    e = jnp.clip(edge, 0, N_EDGES - 1)
    player = state.current_player.astype(jnp.int32)
    use_free = state.free_roads > 0
    new_free = to_u8(
        jnp.where(
            use_free,
            state.free_roads.astype(jnp.int32) - 1,
            state.free_roads.astype(jnp.int32),
        )
    )
    paid = pay(state.player_resources, player, ROAD_COST_ARR)
    new_res = jnp.where(use_free, state.player_resources, paid)
    cand = state._replace(
        edge_road=state.edge_road.at[e].set((player + 1).astype(jnp.uint8)),
        free_roads=new_free,
        player_resources=new_res,
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_build_road_avail_b = jax.jit(jax.vmap(_build_road_avail))
_build_road_apply_b = jax.jit(jax.vmap(_build_road_apply))


def build_road_available(board: Board, edge: IndexParam) -> Mask:
    """``(batch,)`` legality of BuildRoad on ``edge`` per game (no state change)."""
    return cast(Mask, _build_road_avail_b(board[0], board[1], edge))


def build_road_step(board: Board, edge: IndexParam) -> tuple[BoardState, ResultCode]:
    """Apply BuildRoad on ``edge`` per game. Free if free_roads > 0.

    Resolves the Longest Road award and any win via :func:`awards.resolve_step`.
    """
    available = _build_road_avail_b(board[0], board[1], edge)
    state, result = _build_road_apply_b(board[0], board[1], edge, available)
    return cast("tuple[BoardState, ResultCode]", awards.resolve_step_b(state, result))


# ===========================================================================
# BuildSettlement
# ===========================================================================


def _build_settlement_avail(
    layout: BoardLayout, state: BoardState, vertex: IntScalar
) -> BoolScalar:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    main = main_after_roll(state)
    under_max = (
        count_settlements(state.vertex_owner, state.vertex_type, player)
        < MAX_SETTLEMENTS
    )
    afford = can_afford(state.player_resources[player], SETTLEMENT_COST_ARR)
    dist = distance_rule_ok(state.vertex_owner, v)
    conn = settlement_connected(state.edge_road, player, v)
    return in_range & main & under_max & afford & dist & conn


def _build_settlement_apply(
    layout: BoardLayout, state: BoardState, vertex: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    cand = state._replace(
        player_resources=pay(
            state.player_resources, player, SETTLEMENT_COST_ARR
        ),
        vertex_owner=state.vertex_owner.at[v].set((player + 1).astype(jnp.uint8)),
        vertex_type=state.vertex_type.at[v].set(SETTLEMENT),
        victory_points=state.victory_points.at[player].add(1),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_build_settlement_avail_b = jax.jit(jax.vmap(_build_settlement_avail))
_build_settlement_apply_b = jax.jit(jax.vmap(_build_settlement_apply))


def build_settlement_available(board: Board, vertex: IndexParam) -> Mask:
    """``(batch,)`` legality of BuildSettlement on ``vertex`` (no state change)."""
    return cast(Mask, _build_settlement_avail_b(board[0], board[1], vertex))


def build_settlement_step(
    board: Board, vertex: IndexParam
) -> tuple[BoardState, ResultCode]:
    """Apply BuildSettlement on ``vertex`` per game.

    Resolves the Longest Road award (a settlement can cut an opponent's road) and
    any win via :func:`awards.resolve_step`.
    """
    available = _build_settlement_avail_b(board[0], board[1], vertex)
    state, result = _build_settlement_apply_b(board[0], board[1], vertex, available)
    return cast("tuple[BoardState, ResultCode]", awards.resolve_step_b(state, result))


# ===========================================================================
# BuildCity
# ===========================================================================


def _build_city_avail(
    layout: BoardLayout, state: BoardState, vertex: IntScalar
) -> BoolScalar:
    in_range = (vertex >= 0) & (vertex < N_VERTICES)
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    main = main_after_roll(state)
    under_max = (
        count_cities(state.vertex_owner, state.vertex_type, player) < MAX_CITIES
    )
    owns_settlement = (state.vertex_owner[v] == (player + 1).astype(jnp.uint8)) & (
        state.vertex_type[v] == SETTLEMENT
    )
    afford = can_afford(state.player_resources[player], CITY_COST_ARR)
    return in_range & main & under_max & owns_settlement & afford


def _build_city_apply(
    layout: BoardLayout, state: BoardState, vertex: IntScalar, available: BoolScalar
) -> tuple[BoardState, IntScalar]:
    v = jnp.clip(vertex, 0, N_VERTICES - 1)
    player = state.current_player.astype(jnp.int32)
    cand = state._replace(
        player_resources=pay(state.player_resources, player, CITY_COST_ARR),
        vertex_type=state.vertex_type.at[v].set(CITY),
        victory_points=state.victory_points.at[player].add(1),
    )
    return tree_select(available, cand, state), jnp.where(
        available, SUCCESS, INVALID
    )


_build_city_avail_b = jax.jit(jax.vmap(_build_city_avail))
_build_city_apply_b = jax.jit(jax.vmap(_build_city_apply))


def build_city_available(board: Board, vertex: IndexParam) -> Mask:
    """``(batch,)`` legality of BuildCity on ``vertex`` (no state change)."""
    return cast(Mask, _build_city_avail_b(board[0], board[1], vertex))


def build_city_step(board: Board, vertex: IndexParam) -> tuple[BoardState, ResultCode]:
    """Apply BuildCity (upgrade an own settlement) on ``vertex`` per game.

    Resolves any win (the +1 VP can reach the threshold) via
    :func:`awards.resolve_step`.
    """
    available = _build_city_avail_b(board[0], board[1], vertex)
    state, result = _build_city_apply_b(board[0], board[1], vertex, available)
    return cast("tuple[BoardState, ResultCode]", awards.resolve_step_b(state, result))
