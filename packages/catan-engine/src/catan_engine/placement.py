"""Settlement / road placement legality (single-game, traceable)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from catan_engine.geometry import EDGE_V, NO_IDX, V_EDGES, V_NBR


def distance_rule_ok(vertex_owner: jax.Array, vertex: jax.Array) -> jax.Array:
    """Vertex empty and no adjacent vertex carries a building."""
    nbr = V_NBR[vertex]  # (MAX_VERTEX_DEGREE,)
    valid = nbr != NO_IDX
    occ = vertex_owner[jnp.where(valid, nbr, 0)]
    blocked = jnp.any(valid & (occ != 0))
    return (vertex_owner[vertex] == 0) & ~blocked


def settlement_connected(
    edge_road: jax.Array, player: jax.Array, vertex: jax.Array
) -> jax.Array:
    """Player owns a road incident to ``vertex`` (required outside setup)."""
    e = V_EDGES[vertex]
    valid = e != NO_IDX
    roads = edge_road[jnp.where(valid, e, 0)]
    return jnp.any(valid & (roads == player + 1))


def road_placeable(
    edge_road: jax.Array, vertex_owner: jax.Array, player: jax.Array, edge: jax.Array
) -> jax.Array:
    """Edge empty and connects to the player's network at a non-blocked end."""
    target = player + 1
    empty = edge_road[edge] == 0

    def end_ok(v: jax.Array) -> jax.Array:
        own_here = vertex_owner[v] == target
        blocked = (vertex_owner[v] != 0) & ~own_here  # opponent building blocks
        e2 = V_EDGES[v]
        valid = (e2 != NO_IDX) & (e2 != edge)
        roads = edge_road[jnp.where(valid, e2, 0)]
        has_own_adj = jnp.any(valid & (roads == target))
        return own_here | (~blocked & has_own_adj)

    return empty & (end_ok(EDGE_V[edge, 0]) | end_ok(EDGE_V[edge, 1]))
