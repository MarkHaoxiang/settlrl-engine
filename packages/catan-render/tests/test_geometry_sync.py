"""Geometry sync: the renderer's index tables must match the engine's.

``convert.py`` reconstructs the engine's tile / vertex / edge / port index
ordering from scratch (the engine's maps are private). The engine *also* exposes
authoritative host-side index<->cube lookups in ``catan_engine.board.layout``.
If the engine ever reindexes its board, these reconstructions would silently
point at the wrong coordinates — so we pin them index-by-index against the
engine's own lookups here.
"""

from catan_engine.board.layout import (
    N_EDGES,
    N_PORTS,
    N_TILES,
    N_VERTICES,
    PORT_V,
    edge_cubes,
    tile_cube,
    vertex_cube,
)

from catan_render.convert import (
    EDGE_VERTICES,
    PORT_VERTEX_COORDS,
    TILE_COORDS,
    VERTEX_COORDS,
    _TILE_CUBES,
)


def test_counts_match_engine() -> None:
    assert len(_TILE_CUBES) == N_TILES
    assert len(TILE_COORDS) == N_TILES
    assert len(VERTEX_COORDS) == N_VERTICES
    assert len(EDGE_VERTICES) == N_EDGES
    assert len(PORT_VERTEX_COORDS) == N_PORTS


def test_vertex_index_order_matches_engine() -> None:
    # Renderer vertex index i must be the engine's vertex index i.
    for i in range(N_VERTICES):
        assert VERTEX_COORDS[i] == vertex_cube(i), f"vertex {i}"


def test_tile_index_order_matches_engine() -> None:
    for i in range(N_TILES):
        assert _TILE_CUBES[i] == tile_cube(i), f"tile cube {i}"
        # TILE_COORDS is the axial (q, r) projection (cube with s dropped).
        assert TILE_COORDS[i] == tile_cube(i)[:2], f"tile axial {i}"


def test_edge_index_order_matches_engine() -> None:
    # Each renderer edge resolves (via its two vertex indices) to the same pair
    # of cube coordinates the engine stores for that edge index.
    for e in range(N_EDGES):
        v1, v2 = EDGE_VERTICES[e]
        rendered = {VERTEX_COORDS[v1], VERTEX_COORDS[v2]}
        engine = set(edge_cubes(e))
        assert rendered == engine, f"edge {e}"


def test_port_vertices_match_engine() -> None:
    # PORT_V maps port index -> its two coastal vertex indices.
    for p in range(N_PORTS):
        engine = {vertex_cube(int(PORT_V[p][0])), vertex_cube(int(PORT_V[p][1]))}
        rendered = set(PORT_VERTEX_COORDS[p])
        assert rendered == engine, f"port {p}"
