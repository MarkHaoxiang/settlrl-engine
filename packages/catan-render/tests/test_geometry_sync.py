"""Geometry smoke check for the renderer's coordinate tables.

``convert.py`` now builds its tile / vertex / edge / port coordinate tables
directly from the engine's authoritative host-side lookups
(``vertex_cube`` / ``edge_cubes`` / ``tile_cube`` / ``PORT_V``), so they can no
longer drift from the engine. This test just sanity-checks shapes/counts and
that the derived tables are internally consistent (no stray placeholders).
"""

from catan_engine.board.layout import N_EDGES, N_PORTS, N_TILES, N_VERTICES
from catan_render.convert import (
    _TILE_CUBES,
    EDGE_VERTICES,
    PORT_VERTEX_COORDS,
    TILE_COORDS,
    VERTEX_COORDS,
)


def test_counts_match_engine() -> None:
    assert len(_TILE_CUBES) == N_TILES
    assert len(TILE_COORDS) == N_TILES
    assert len(VERTEX_COORDS) == N_VERTICES
    assert len(EDGE_VERTICES) == N_EDGES
    assert len(PORT_VERTEX_COORDS) == N_PORTS


def test_tables_are_well_formed() -> None:
    # Tile axial coords are the (q, r) projection of the tile cube centres.
    for (q, r), cube in zip(TILE_COORDS, _TILE_CUBES, strict=True):
        assert (q, r) == cube[:2]
    # Edges reference valid vertex indices (and aren't self-loops).
    for v1, v2 in EDGE_VERTICES:
        assert 0 <= v1 < N_VERTICES and 0 <= v2 < N_VERTICES
        assert v1 != v2
    # Vertices are cube triples summing to +/-1; ports carry two distinct ones.
    for cube in VERTEX_COORDS:
        assert sum(cube) in (1, -1)
    for a, b in PORT_VERTEX_COORDS:
        assert a != b
