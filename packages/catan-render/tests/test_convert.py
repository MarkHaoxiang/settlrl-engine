"""Unit tests for :func:`convert.board_to_model`.

Build a small engine board with a *known* settlement, city, road, robber and
hand, convert it, and assert the resulting :class:`BoardModel` fields exactly —
the player-index mapping (engine stores player+1, 0 = empty), the settlement vs
city distinction, the per-resource / per-dev-card breakdown, the robber tile,
and the general (3:1) port's ``resource is None``.
"""

from catan_engine.board import (
    Board,
    give,
    give_dev_card,
    make_board,
    place_city,
    place_road,
    place_settlement,
    set_robber,
)
from catan_engine.board.dev_cards import DEV_CARD_COUNTS, DevCard
from catan_engine.board.layout import edge_index, tile_index, vertex_cube, vertex_index
from catan_engine.board.port import Port
from catan_render.convert import _DEV_CARD_NAMES, _RESOURCE_NAMES, board_to_model

# Known cube coordinates to build on (all valid vertices / edge / tile).
SETTLEMENT_CUBE = (1, 0, 0)
CITY_CUBE = (-1, 0, 0)
ROAD_A = (1, 0, 0)
ROAD_B = (0, 0, -1)
ROBBER_CUBE = (0, 0, 0)


def _build_board() -> Board:
    board = make_board(batch_size=1, seed=0)
    # Seat 0 settlement, seat 1 city, seat 0 road.
    board = place_settlement(board, player=0, vertex=vertex_index(SETTLEMENT_CUBE))
    board = place_city(board, player=1, vertex=vertex_index(CITY_CUBE))
    board = place_road(board, player=0, edge=edge_index(ROAD_A, ROAD_B))
    # Seat 0 hand: [sheep, wheat, wood, brick, ore].
    board = give(board, player=0, resources=[1, 2, 3, 4, 5])
    # Seat 0 dev cards: 2 knights, 1 monopoly.
    board = give_dev_card(board, player=0, card=int(DevCard.KNIGHT), count=2)
    board = give_dev_card(board, player=0, card=int(DevCard.MONOPOLY), count=1)
    # Robber to a known tile.
    board = set_robber(board, tile=tile_index(ROBBER_CUBE))
    return board


def test_buildings_owner_mapping_and_kind() -> None:
    model = board_to_model(_build_board())

    # Only the two occupied vertices appear; owner == 0 (empty) is skipped.
    assert len(model.buildings) == 2
    by_coord = {(b.cube.q, b.cube.r, b.cube.s): b for b in model.buildings}

    settle = by_coord[SETTLEMENT_CUBE]
    assert settle.player == 0  # engine player+1 == 1 -> 0-indexed 0
    assert settle.kind == "settlement"

    city = by_coord[CITY_CUBE]
    assert city.player == 1  # engine player+1 == 2 -> 0-indexed 1
    assert city.kind == "city"


def test_road_owner_and_endpoints() -> None:
    model = board_to_model(_build_board())
    assert len(model.roads) == 1
    road = model.roads[0]
    assert road.player == 0
    endpoints = {(road.a.q, road.a.r, road.a.s), (road.b.q, road.b.r, road.b.s)}
    assert endpoints == {ROAD_A, ROAD_B}


def test_player_resource_and_dev_breakdown() -> None:
    model = board_to_model(_build_board())
    p0 = model.players[0]

    # Per-resource breakdown in enum order: sheep=1, wheat=2, wood=3, brick=4, ore=5.
    expected_res = dict(zip(_RESOURCE_NAMES, [1, 2, 3, 4, 5], strict=True))
    res = p0.resources.model_dump()
    assert {k: res[k] for k in _RESOURCE_NAMES} == expected_res
    assert p0.resource_cards == sum(expected_res.values()) == 15

    # Dev cards: 2 knight, 1 monopoly, rest 0.
    dev = p0.dev_card_types.model_dump()
    assert dev["knight"] == 2
    assert dev["monopoly"] == 1
    assert dev["road_building"] == 0
    assert dev["year_of_plenty"] == 0
    assert dev["victory_point"] == 0
    assert p0.dev_cards == 3
    # Field set matches the dev-card enum ordering used to fill it.
    assert tuple(dev) == _DEV_CARD_NAMES


def test_bank_mirrors_hands() -> None:
    model = board_to_model(_build_board())
    assert model.bank is not None
    # Seat 0 holds [1, 2, 3, 4, 5]; the bank holds 19 minus what's in hands.
    bank = model.bank.resources.model_dump()
    expected = dict(zip(_RESOURCE_NAMES, [18, 17, 16, 15, 14], strict=True))
    assert {k: bank[k] for k in _RESOURCE_NAMES} == expected

    fresh = board_to_model(make_board(batch_size=1, seed=0))
    assert fresh.bank is not None
    assert fresh.bank.dev_cards == sum(DEV_CARD_COUNTS)


def test_robber_coordinate() -> None:
    model = board_to_model(_build_board())
    assert model.robber is not None
    assert (model.robber.q, model.robber.r) == ROBBER_CUBE[:2]


def test_general_port_has_no_resource() -> None:
    board = _build_board()
    model = board_to_model(board)
    _, _ = board  # layout/state
    layout = board[0]
    port_alloc = layout.port_allocation[0]

    saw_general = False
    for i, port in enumerate(model.ports):
        if int(port_alloc[i]) == int(Port.GENERAL):
            assert port.resource is None
            saw_general = True
        else:
            assert port.resource is not None
            assert port.resource == Port(int(port_alloc[i])).name.lower()
    assert saw_general, "expected at least one 3:1 general port"


def test_vertex_coords_are_engine_cubes() -> None:
    # A converted building's cube is exactly the engine's vertex_cube for that idx.
    model = board_to_model(_build_board())
    settle = next(b for b in model.buildings if b.player == 0)
    assert (settle.cube.q, settle.cube.r, settle.cube.s) == vertex_cube(
        vertex_index(SETTLEMENT_CUBE)
    )
