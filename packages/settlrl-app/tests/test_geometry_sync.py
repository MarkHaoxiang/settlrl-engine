"""Geometry smoke check for the app's coordinate tables.

``convert.py`` builds its tile / vertex / edge coordinate tables from
``settlrl_game.reference.board``'s cube lookups, so they can no longer drift. This
sanity-checks counts and internal consistency (no stray placeholders).
"""

from settlrl_game.convert import EDGE_VERTICES, TILE_COORDS, VERTEX_COORDS
from settlrl_game.reference import board as rb


def test_tables_are_well_formed() -> None:
    # Tile axial coords are the (q, r) projection of the tile cube centres.
    for t, (q, r) in enumerate(TILE_COORDS):
        cq, cr, _ = rb.tile_cube(t)
        assert (q, r) == (cq, cr)
    # Vertices are distinct cube triples summing to +/-1.
    assert len(set(VERTEX_COORDS)) == rb.N_VERTICES
    for cube in VERTEX_COORDS:
        assert sum(cube) in (1, -1)
    # Every edge joins two real, distinct vertices.
    for a, b in EDGE_VERTICES:
        assert 0 <= a < rb.N_VERTICES and 0 <= b < rb.N_VERTICES and a != b
