"""Enum / ordering sync: the wire models read reference hands positionally, so
their field order must match the reference ``Resource`` / ``DevCard`` enums, and
every terrain must have a wire ``Terrain`` member. Pin them here.
"""

import settlrl_reference as ref
from settlrl_render.api.convert import _RESOURCE_NAMES
from settlrl_render.api.models import DevCardCounts, ResourceCounts, Terrain

# The five resources, in reference (Resource) order.
_RESOURCES = tuple(r.name.lower() for r in ref.RESOURCES)


def test_resource_names_match_enum() -> None:
    # actions.py labels (monopoly / year-of-plenty / maritime trade) use this.
    assert _RESOURCE_NAMES == _RESOURCES


def test_resource_counts_fields_match_enum() -> None:
    # ResourceCounts must read hands in Resource order (sheep..ore).
    assert tuple(ResourceCounts.model_fields) == _RESOURCES


def test_dev_card_counts_fields_match_enum() -> None:
    # DevCardCounts must read dev hands in DevCard order.
    assert tuple(DevCardCounts.model_fields) == tuple(
        d.name.lower() for d in ref.DevCard
    )


def test_terrain_covers_resources_and_desert() -> None:
    rendered = {m.value for m in Terrain}
    assert rendered == {r.name.lower() for r in ref.RESOURCES} | {"desert"}
