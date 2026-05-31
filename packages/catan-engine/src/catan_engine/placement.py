"""Settlement / road placement legality (single-game, traceable).

Vertex incidence is computed by scattering per-edge values onto their endpoints
over the COO ``edge_index`` (``EDGE_V``), rather than gathering through padded
vertex->edge reverse maps -- the JAX analogue of PyG message passing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.layout import EDGE_V, N_VERTICES
from catan_engine.state import BoolScalar, EdgeRoadVec, IntScalar, VertexOwnerVec

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
