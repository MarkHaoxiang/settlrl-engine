"""Flat action table: the renderer's own enumeration round-trips and decodes.

The renderer assigns each concrete action a flat index (``api.actions``). Every
row must decode to an :class:`ActionModel` carrying the right geometry, and
``to_action`` / ``flat_for_action`` must invert each other so a move chosen by
flat id reconstructs the same reference action. Driven over a real game so only
genuinely reachable actions are exercised for legality.
"""

from random import Random

import settlrl_game.reference as ref
from settlrl_game.reference import board as rb
from settlrl_game.actions import (
    N_FLAT,
    decode_actions,
    flat_for_action,
    to_action,
)
from settlrl_game.convert import _RESOURCE_NAMES
from settlrl_game.session import GameSession

ALL_FLAT = list(range(N_FLAT))
_VERTEX_TYPES = {"setup_settlement", "build_settlement", "build_city"}
_ROAD_TYPES = {"setup_road", "build_road"}
_ROBBER_TYPES = {"move_robber", "play_knight"}


def test_every_flat_index_decodes() -> None:
    models = decode_actions(ALL_FLAT)
    assert len(models) == N_FLAT
    for flat, m in enumerate(models):
        assert m.flat == flat
        assert m.label  # every action carries a human label


def test_decoded_geometry_matches_reference_lookups() -> None:
    game = ref.Game.new(ref.random_layout(Random(0)), 0, n_players=4)
    for flat in ALL_FLAT:
        m = decode_actions([flat])[0]
        action = to_action(flat, game)
        if m.type in _VERTEX_TYPES:
            assert m.vertex is not None
            v = action.vertex  # type: ignore[union-attr]
            assert (m.vertex.q, m.vertex.r, m.vertex.s) == rb.vertex_cube(v)
        elif m.type in _ROAD_TYPES:
            assert m.edge is not None
            rendered = {
                (m.edge.a.q, m.edge.a.r, m.edge.a.s),
                (m.edge.b.q, m.edge.b.r, m.edge.b.s),
            }
            va, vb = rb.edge_vertices(action.edge)  # type: ignore[union-attr]
            assert rendered == {rb.vertex_cube(va), rb.vertex_cube(vb)}
        elif m.type in _ROBBER_TYPES:
            assert m.tile is not None
            assert (m.tile.q, m.tile.r) == rb.tile_cube(action.tile)[:2]  # type: ignore[union-attr]


def test_flat_round_trips_over_a_random_game() -> None:
    # Every legal flat reconstructs a legal reference action, and maps back to
    # the same flat id.
    sess = GameSession(seed=0, n_players=4)
    for _ in range(500):
        flats = sess.legal_flat()
        if not flats:
            break
        for f in flats:
            action = to_action(f, sess.game)
            assert sess.game.is_legal(action)
            assert flat_for_action(action) == f
        if sess.auto_step() is None:
            break


def test_resource_actions_decode_their_resource() -> None:
    for flat in ALL_FLAT:
        m = decode_actions([flat])[0]
        if m.type in ("discard", "play_monopoly"):
            assert m.resource in _RESOURCE_NAMES
        elif m.type == "maritime_trade":
            assert m.give in _RESOURCE_NAMES and m.receive in _RESOURCE_NAMES
