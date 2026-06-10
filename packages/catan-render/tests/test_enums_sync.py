"""Enum / ordering sync: the renderer hard-codes resource, dev-card and terrain
orderings that mirror the engine's enums. The engine indexes resource arrays by
the ``Tile`` enum (0-4) and dev-card arrays by the ``DevCard`` enum, so if those
enums are reordered the renderer would mislabel hands. Pin them here.
"""

from catan_engine.board.dev_cards import DevCard
from catan_engine.board.tile import Tile
from catan_render.actions import _RESOURCE_NAMES
from catan_render.models import DevCardCounts, ResourceCounts, Terrain

# The five non-desert resources, in engine (Tile) order.
_RESOURCES = tuple(t.name.lower() for t in Tile if t is not Tile.DESERT)


def test_resource_names_match_tile_enum() -> None:
    # actions.py labels (monopoly / year-of-plenty / maritime trade) use this.
    assert _RESOURCE_NAMES == _RESOURCES


def test_resource_counts_fields_match_tile_enum() -> None:
    # ResourceCounts must read player_resources in Tile order (sheep..ore).
    assert tuple(ResourceCounts.model_fields) == _RESOURCES


def test_dev_card_counts_fields_match_dev_card_enum() -> None:
    # DevCardCounts must read dev_hand in DevCard order.
    expected = tuple(d.name.lower() for d in DevCard)
    assert tuple(DevCardCounts.model_fields) == expected


def test_terrain_covers_every_tile() -> None:
    # Every engine terrain (incl. desert) must have a wire Terrain member.
    engine = {t.name.lower() for t in Tile}
    rendered = {m.value for m in Terrain}
    assert rendered == engine
