"""Win timing (rulebook p.5, "Ending the Game").

You can only win *during your own turn*. A player who reaches 10 VP out of
turn (e.g. a settlement break handing them Longest Road) keeps waiting; play
continues until any player is at 10 VP on their own turn, which they claim at
the turn's start.
"""

from __future__ import annotations

from conftest import make_game, make_layout, place
from settlrl_game.reference.game import EndTurn, Game, Roll
from settlrl_game.reference.types import Building, Phase


def _give_ten_vp(game: Game, player: int) -> None:
    """Ten victory points of buildings for ``player`` (3 cities + 4
    settlements, within the piece caps), on otherwise-unused vertices."""
    base = 20 + 10 * player  # vertex blocks far apart per player
    for v in range(base, base + 3):
        place(game, v, player, Building.CITY)
    for v in range(base + 3, base + 7):
        place(game, v, player, Building.SETTLEMENT)


def test_off_turn_ten_vp_claims_at_turn_start() -> None:
    game = make_game(make_layout())  # player 0's MAIN, has_rolled
    _give_ten_vp(game, 1)
    assert game.total_vp(1) == 10
    assert game.phase is Phase.MAIN  # 10 VP out of turn ends nothing

    game.apply(EndTurn())  # player 1's turn begins -> they claim

    assert game.phase is Phase.GAME_OVER
    assert game.current_player == 1


def test_game_continues_until_the_ten_vp_players_turn() -> None:
    game = make_game(make_layout())
    _give_ten_vp(game, 2)

    game.apply(EndTurn())  # player 1's turn: not the 10-VP player
    assert game.phase is Phase.ROLL
    assert game.current_player == 1

    game.apply(Roll(value=4))  # all-desert layout: no production, no 7
    game.apply(EndTurn())  # player 2's turn begins -> they claim

    assert game.phase is Phase.GAME_OVER
    assert game.current_player == 2
