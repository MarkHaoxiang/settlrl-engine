"""Persistence: journals on the shared async DB replay back into live games on
restart. The write-behind store is exercised through the HTTP surface (a clean
``with`` shutdown drains queued writes); the eviction unit test drives the store
and registry directly under ``asyncio.run``.
"""

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient
from settlrl_render.game.games import GameRegistry
from settlrl_render.game.session import GameSession, GameSetup
from settlrl_render.server import create_app
from settlrl_render.storage.db import Database
from settlrl_render.storage.store import GameStore


def test_game_setup_round_trips_through_a_dict() -> None:
    setup = GameSetup(
        seed=3,
        n_players=2,
        number_placement="spiral",
        seats=["human", {"kind": "random", "params": {}}],
    )
    # from_dict ignores the journal's framing keys (id, t).
    assert GameSetup.from_dict({**setup.to_dict(), "id": "x", "t": "header"}) == setup


def test_session_setup_captures_its_seats() -> None:
    session = GameSession(seed=5, n_players=2, seats=["human", "random"])
    assert session.setup == GameSetup(5, 2, "random", ["human", "random"])


def test_restart_resumes_an_in_progress_game(tmp_path: Path) -> None:
    # First boot: create a game (with a bot seat), play a move, leave a chat line.
    with TestClient(create_app(state_dir=str(tmp_path), warm=False)) as c1:
        doc = c1.post(
            "/api/games",
            json={
                "seed": 0,
                "n_players": 2,
                "seats": ["human", "random"],
                "claim": "first",
            },
        ).json()
        game = doc["id"]
        hdr = {"X-Seat-Tokens": ",".join(dict(doc["tokens"]).values())}
        flat = c1.get(f"/api/games/{game}", headers=hdr).json()["actions"][0]["flat"]
        before = c1.post(
            f"/api/games/{game}/action", json={"flat": flat}, headers=hdr
        ).json()
        c1.post(
            f"/api/games/{game}/chat", json={"text": "gg", "player": 0}, headers=hdr
        )

    # Second boot from the same dir: the game is back, at the same position.
    with TestClient(create_app(state_dir=str(tmp_path), warm=False)) as c2:
        after = c2.get(f"/api/games/{game}", headers=hdr).json()
        assert after["id"] == game
        assert after["board"] == before["board"]  # replayed to the same state
        assert after["seats_claimed"] == [0]  # the claim (and its kinds) survived
        assert after["status"]["seats"] == ["human", "random"]
        assert any(e["kind"] == "chat" and e["text"] == "gg" for e in after["log"])


def test_eviction_drops_the_game_from_the_store(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, str, set[str]]:
        db = Database(str(tmp_path / "settlrl.db"))
        await db.init()
        store = GameStore(db)
        store.start()
        reg = GameRegistry(max_games=1, store=store)
        a = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
        # Idle well past the TTL (relative to monotonic(), which need not be large).
        a.touched = time.monotonic() - 100_000
        b = reg.create(GameSession(seed=1, n_players=2, seats=["human", "human"]))
        await store.aclose()  # flush the header writes and the eviction removal
        stored = {str(header["id"]) for header, _ in await store.load()}
        await db.dispose()
        return a.id, b.id, stored

    a_id, b_id, stored = asyncio.run(scenario())
    assert a_id not in stored and b_id in stored


def test_restored_bot_game_resumes_playing(tmp_path: Path) -> None:
    # First boot: an all-bot game plays a few moves, journalled, then we shut
    # down cleanly (draining the queued writes).
    with TestClient(
        create_app(state_dir=str(tmp_path), bot_delay=0.0, warm=False)
    ) as c1:
        game = c1.post(
            "/api/games",
            json={
                "seed": 0,
                "n_players": 2,
                "seats": ["random", "random"],
                "claim": "none",
            },
        ).json()["id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            snap = c1.get(f"/api/games/{game}").json()
            moves = sum(1 for e in snap["log"] if e["kind"] == "move")
            if snap["status"]["terminal"] or moves >= 5:
                break
            time.sleep(0.05)

    # A fresh app restores the position and its startup restarts the driver,
    # which plays the game out to the end.
    with TestClient(
        create_app(state_dir=str(tmp_path), bot_delay=0.0, warm=False)
    ) as c2:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if c2.get(f"/api/games/{game}").json()["status"]["terminal"]:
                break
            time.sleep(0.1)
        body = c2.get(f"/api/games/{game}").json()
        assert body["status"]["terminal"] and body["status"]["winner"] is not None
