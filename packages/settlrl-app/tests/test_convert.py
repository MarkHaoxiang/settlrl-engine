"""Unit tests for :func:`convert.board_to_model`.

Build a small reference game with a *known* settlement, city, road, robber and
hand, convert it, and assert the resulting :class:`BoardModel` fields exactly —
the settlement vs city distinction, the per-resource / per-dev-card breakdown,
the robber tile, and the general (3:1) port's ``resource is None``.
"""

from random import Random

import settlrl_game.reference as ref
from settlrl_game.convert import _DEV_CARD_NAMES, _RESOURCE_NAMES, board_to_model
from settlrl_game.reference import board as rb
from settlrl_game.reference.types import DEV_CARD_COUNTS, Building, DevCard

# Known cube coordinates to build on (all valid vertices / edge / tile).
SETTLEMENT_CUBE = (1, 0, 0)
CITY_CUBE = (-1, 0, 0)
ROAD_A = (1, 0, 0)
ROAD_B = (0, 0, -1)
ROBBER_CUBE = (0, 0, 0)


def _build_game() -> ref.Game:
    layout = ref.random_layout(Random(0))
    game = ref.Game.new(layout, ref.desert_tile(layout), n_players=4)
    game.buildings[rb.cube_to_vertex(SETTLEMENT_CUBE)] = (0, Building.SETTLEMENT)
    game.buildings[rb.cube_to_vertex(CITY_CUBE)] = (1, Building.CITY)
    edge = rb.edge_between(rb.cube_to_vertex(ROAD_A), rb.cube_to_vertex(ROAD_B))
    game.roads[edge] = 0
    for resource, n in zip(ref.RESOURCES, [1, 2, 3, 4, 5], strict=True):
        game.players[0].resources[resource] = n
    game.players[0].dev_cards[DevCard.KNIGHT] = 2
    game.players[0].dev_cards[DevCard.MONOPOLY] = 1
    game.robber = rb.cube_to_tile(ROBBER_CUBE)
    return game


def test_buildings_owner_mapping_and_kind() -> None:
    model = board_to_model(_build_game())
    assert len(model.buildings) == 2
    by_coord = {(b.cube.q, b.cube.r, b.cube.s): b for b in model.buildings}

    settle = by_coord[SETTLEMENT_CUBE]
    assert settle.player == 0
    assert settle.kind == "settlement"

    city = by_coord[CITY_CUBE]
    assert city.player == 1
    assert city.kind == "city"


def test_road_owner_and_endpoints() -> None:
    model = board_to_model(_build_game())
    assert len(model.roads) == 1
    road = model.roads[0]
    assert road.player == 0
    endpoints = {(road.a.q, road.a.r, road.a.s), (road.b.q, road.b.r, road.b.s)}
    assert endpoints == {ROAD_A, ROAD_B}


def test_player_resource_and_dev_breakdown() -> None:
    p0 = board_to_model(_build_game()).players[0]
    assert p0.resources is not None and p0.dev_card_types is not None
    expected_res = dict(zip(_RESOURCE_NAMES, [1, 2, 3, 4, 5], strict=True))
    res = p0.resources.model_dump()
    assert {k: res[k] for k in _RESOURCE_NAMES} == expected_res
    assert p0.resource_cards == sum(expected_res.values()) == 15

    dev = p0.dev_card_types.model_dump()
    assert dev["knight"] == 2
    assert dev["monopoly"] == 1
    assert dev["road_building"] == 0
    assert dev["year_of_plenty"] == 0
    assert dev["victory_point"] == 0
    assert p0.dev_cards == 3
    assert tuple(dev) == _DEV_CARD_NAMES


def test_bank_mirrors_hands() -> None:
    model = board_to_model(_build_game())
    assert model.bank is not None
    bank = model.bank.resources.model_dump()
    expected = dict(zip(_RESOURCE_NAMES, [18, 17, 16, 15, 14], strict=True))
    assert {k: bank[k] for k in _RESOURCE_NAMES} == expected

    layout = ref.random_layout(Random(0))
    fresh = board_to_model(ref.Game.new(layout, ref.desert_tile(layout)))
    assert fresh.bank is not None
    assert fresh.bank.dev_cards == sum(DEV_CARD_COUNTS.values())


def test_robber_coordinate() -> None:
    model = board_to_model(_build_game())
    assert model.robber is not None
    assert (model.robber.q, model.robber.r) == ROBBER_CUBE[:2]


def test_general_port_has_no_resource() -> None:
    game = _build_game()
    model = board_to_model(game)
    saw_general = False
    for wire, port in zip(model.ports, game.layout.ports, strict=True):
        if port.type is ref.PortType.GENERIC:
            assert wire.resource is None
            saw_general = True
        else:
            assert wire.resource == port.type.value.name.lower()
    assert saw_general, "expected at least one 3:1 general port"


def test_vertex_coords_are_cubes() -> None:
    model = board_to_model(_build_game())
    settle = next(b for b in model.buildings if b.player == 0)
    assert (settle.cube.q, settle.cube.r, settle.cube.s) == SETTLEMENT_CUBE
