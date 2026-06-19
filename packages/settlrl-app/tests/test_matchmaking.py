"""Elo Quick Match: pairing near-rated waiters and bot-filling the rest.

The :class:`Matchmaker` is exercised directly (a fake clock makes the never-stuck
timeout deterministic) and once over HTTP to check the route wiring.
"""

import asyncio
from collections.abc import Iterator

import pytest
from _helpers import bot_registry
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.game.matchmaking import Matchmaker, elo_window
from settlrl_app.server import create_app
from settlrl_game.session import HUMAN


def _matchmaker(**kw: object) -> tuple[Matchmaker, GameRegistry]:
    registry = GameRegistry()
    mm = Matchmaker(registry, bot_registry(), lambda h: None, **kw)  # type: ignore[arg-type]
    return mm, registry


def test_elo_window_widens_with_wait() -> None:
    assert elo_window(0, 150, 75) == 150
    assert elo_window(10, 150, 75) == 225
    assert elo_window(20, 150, 75) == 300


def test_two_near_rated_waiters_pair_into_one_game() -> None:
    async def scenario() -> None:
        mm, registry = _matchmaker()
        a = await mm.matchmake(2, None, None)
        assert a.result is None and a.waiting == 1  # alone: still in line

        b = await mm.matchmake(2, None, None)  # second waiter completes the table
        assert b.result is not None

        a2 = await mm.matchmake(2, a.ticket, None)  # first picks up its seat
        assert a2.result is not None
        assert a2.result[0] == b.result[0]  # the same game
        assert {a2.result[1], b.result[1]} == {0, 1}  # distinct seats

        handle = registry.get(b.result[0])
        assert handle is not None
        assert handle.session.seats == [HUMAN, HUMAN] and handle.ready()

    asyncio.run(scenario())


def test_lone_waiter_is_bot_filled_once_it_times_out() -> None:
    async def scenario() -> None:
        # never_stuck_s=0: a single waiter forms a game immediately, bots filling
        # the empty seat (its rating closest to the human's).
        mm, registry = _matchmaker(never_stuck_s=0.0)
        a = await mm.matchmake(2, None, None)
        assert a.result is not None

        handle = registry.get(a.result[0])
        assert handle is not None
        assert handle.session.seats[0] == HUMAN
        assert handle.session.seats[1] != HUMAN  # a bot fills the rest
        assert handle.ready()

    asyncio.run(scenario())


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app(GameRegistry(), providers=bot_registry())) as c:
        yield c


def test_matchmake_route_pairs_two_callers(client: TestClient) -> None:
    first = client.post("/api/matchmake", json={"n_players": 2}).json()
    assert first["queued"] is True and first["waiting"] == 1

    second = client.post("/api/matchmake", json={"n_players": 2}).json()
    assert "id" in second  # the second caller completes and gets a seat

    resumed = client.post(
        "/api/matchmake", json={"n_players": 2, "ticket": first["ticket"]}
    ).json()
    assert resumed["id"] == second["id"]  # both land in the same game
    assert {resumed["seat"], second["seat"]} == {0, 1}
