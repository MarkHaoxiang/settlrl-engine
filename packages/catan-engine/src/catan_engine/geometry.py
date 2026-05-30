"""Static board geometry as int32 ``jnp`` arrays for traceable gather/index.

These mirror the padded incidence maps generated in ``layout.py`` but cast to
``int32`` so the single-game rule helpers can gather through them under a JAX
trace. ``NO_INDEX`` padding is exposed here as ``NO_IDX`` (a 0-d ``int32``); rule
code handles it by masking + clipped gathers rather than Python branching.
"""

from __future__ import annotations

import jax.numpy as jnp

from catan_engine.layout import (
    NO_INDEX,
    _edge_vertex_map,
    _tile_vertex_map,
    _vertex_edge_map,
    _vertex_neighbour_map,
    _vertex_port_map,
    _vertex_tile_map,
)

EDGE_V = _edge_vertex_map.astype(jnp.int32)  # (N_EDGES, 2)
V_EDGES = _vertex_edge_map.astype(jnp.int32)  # (N_VERTICES, MAX_VERTEX_DEGREE)
V_NBR = _vertex_neighbour_map.astype(jnp.int32)  # (N_VERTICES, MAX_VERTEX_DEGREE)
V_TILES = _vertex_tile_map.astype(jnp.int32)  # (N_VERTICES, MAX_VERTEX_DEGREE)
V_PORT = _vertex_port_map.astype(jnp.int32)  # (N_VERTICES,)
TILE_V = _tile_vertex_map.astype(jnp.int32)  # (N_TILES, 6)

NO_IDX = jnp.int32(NO_INDEX)
