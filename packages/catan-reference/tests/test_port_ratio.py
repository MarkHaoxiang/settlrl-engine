"""Maritime trade ratio with multiple owned ports (``Game.port_ratio``, p.7/p.8).

A player who controls more than one harbour trades at the **best** (lowest) ratio
any of them offers: 4:1 by default, 3:1 at a generic port, 2:1 at the matching
resource port. The tests give one player buildings on several port vertices and
check the chosen ratio per resource.
"""

from __future__ import annotations

from catan_reference.board import Layout, Port
from catan_reference.types import Building, PortType, Resource
from conftest import make_game, make_layout, place

# Distinct, well-separated vertices to carry the ports (geometry is irrelevant to
# port_ratio, which only looks up the port at each owned vertex).
V_GENERIC = 0
V_ORE = 20
V_BRICK = 40


def _layout_with_ports() -> Layout:
    return make_layout(
        ports=(
            Port(type=PortType.GENERIC, vertices=(V_GENERIC, 5)),
            Port(type=PortType.ORE, vertices=(V_ORE, 21)),
            Port(type=PortType.BRICK, vertices=(V_BRICK, 41)),
        ),
    )


def test_best_ratio_chosen_across_multiple_ports() -> None:
    game = make_game(_layout_with_ports())
    # Player 0 owns the generic (3:1) and ore (2:1) ports.
    place(game, V_GENERIC, player=0, kind=Building.SETTLEMENT)
    place(game, V_ORE, player=0, kind=Building.CITY)

    # Giving ORE: the 2:1 ore port beats the generic 3:1.
    assert game.port_ratio(0, Resource.ORE) == 2
    # Giving WOOD: no 2:1 match, but the generic port still gives 3:1.
    assert game.port_ratio(0, Resource.WOOD) == 3


def test_two_to_one_port_only_helps_its_own_resource() -> None:
    game = make_game(_layout_with_ports())
    # Player 0 owns only the ore 2:1 port.
    place(game, V_ORE, player=0, kind=Building.SETTLEMENT)

    assert game.port_ratio(0, Resource.ORE) == 2
    # Any other resource falls back to the default 4:1.
    assert game.port_ratio(0, Resource.WHEAT) == 4


def test_two_matching_ports_pick_the_two_to_one() -> None:
    # Owning both a generic and the matching 2:1 for BRICK -> the 2:1 wins.
    game = make_game(_layout_with_ports())
    place(game, V_GENERIC, player=0, kind=Building.SETTLEMENT)
    place(game, V_BRICK, player=0, kind=Building.SETTLEMENT)

    assert game.port_ratio(0, Resource.BRICK) == 2
    assert game.port_ratio(0, Resource.ORE) == 3  # only the generic applies here


def test_ports_owned_by_others_do_not_help() -> None:
    game = make_game(_layout_with_ports())
    # The ore port belongs to player 1, not player 0.
    place(game, V_ORE, player=1, kind=Building.SETTLEMENT)

    assert game.port_ratio(0, Resource.ORE) == 4
