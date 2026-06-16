"""Longest Road tie handling (rulebook p.9, ``Game.recompute_longest_road``).

Two rulebook subtleties random play rarely lands on cleanly:

* the current holder **keeps** the card while still tied for the longest road;
* if the longest road is tied with **no prior holder** (the holder having been
  beaten, or never assigned), the card is **set aside** -- no one holds it.

Both players are given disjoint 5-segment roads (``LONGEST_ROAD_MIN``), so the
two longest roads are exactly tied.
"""

from __future__ import annotations

from conftest import make_game, make_layout, place_road_path
from settlrl_reference.game import Game
from settlrl_reference.types import LONGEST_ROAD_MIN

# Two vertex-disjoint paths of 5 edges each (see board.VERTEX_NEIGHBORS).
PATH_P0 = [0, 5, 3, 13, 11, 23]
PATH_P1 = [1, 9, 2, 10, 8, 19]


def _two_tied_roads() -> Game:
    game = make_game(make_layout())
    e0 = place_road_path(game, PATH_P0, player=0)
    e1 = place_road_path(game, PATH_P1, player=1)
    assert len(e0) == LONGEST_ROAD_MIN and len(e1) == LONGEST_ROAD_MIN
    assert game.longest_road_length(0) == LONGEST_ROAD_MIN
    assert game.longest_road_length(1) == LONGEST_ROAD_MIN
    return game


def test_holder_keeps_card_while_tied() -> None:
    game = _two_tied_roads()
    # Player 1 already holds the card; a tie must not strip it from them.
    game.longest_road_owner = 1
    game.longest_road_len = LONGEST_ROAD_MIN

    game.recompute_longest_road()

    assert game.longest_road_owner == 1
    assert game.longest_road_len == LONGEST_ROAD_MIN


def test_tie_with_no_prior_holder_sets_card_aside() -> None:
    game = _two_tied_roads()
    # No one held it before -> a tie for the longest leaves it unheld.
    game.longest_road_owner = None
    game.longest_road_len = 0

    game.recompute_longest_road()

    assert game.longest_road_owner is None
    assert game.longest_road_len == 0


def test_holder_beaten_by_tied_pair_sets_card_aside() -> None:
    # Player 2 used to hold a (now shorter) road; players 0 and 1 tie for a new,
    # strictly longer road. The beaten holder is not among the leaders and the new
    # lead is tied -> the card is set aside.
    game = _two_tied_roads()
    game.longest_road_owner = 2
    game.longest_road_len = 4  # a stale, shorter length

    game.recompute_longest_road()

    assert game.longest_road_owner is None
    assert game.longest_road_len == 0


def test_single_unique_longest_takes_card() -> None:
    # Control: extend player 0's road to 6 so it is the unique longest; the card
    # goes to player 0 even with no prior holder.
    game = make_game(make_layout())
    place_road_path(game, [0, 5, 3, 13, 11, 23, 21], player=0)  # 6 segments
    place_road_path(game, PATH_P1, player=1)  # 5 segments
    assert game.longest_road_length(0) == 6
    assert game.longest_road_length(1) == LONGEST_ROAD_MIN
    game.longest_road_owner = None

    game.recompute_longest_road()

    assert game.longest_road_owner == 0
    assert game.longest_road_len == 6
