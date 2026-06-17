"""Persistence: journals on disk replay back into live games on restart."""

import time
from pathlib import Path

from fastapi.testclient import TestClient
from settlrl_render.games import GameRegistry, restore_registry
from settlrl_render.server import create_app
from settlrl_render.session import GameSession, GameSetup
from settlrl_render.store import GameStore


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
    # First boot: create a game, play a move, leave a chat line.
    with TestClient(create_app(state_dir=str(tmp_path), warm=False)) as c1:
        doc = c1.post(
            "/api/games", json={"seed": 0, "seats": ["human"] * 4, "claim": "first"}
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
        assert after["seats_claimed"] == [0]  # the claim survived
        assert any(e["kind"] == "chat" and e["text"] == "gg" for e in after["log"])
        # The restored token still proves the seat.
        assert after["status"]["seats"][0] == "human"


def test_restore_preserves_seat_kinds(tmp_path: Path) -> None:
    store = GameStore(tmp_path)
    reg = GameRegistry(store=store)
    handle = reg.create(GameSession(seed=0, n_players=2, seats=["human", "random"]))
    _, token = handle.claim(0)

    restored = restore_registry(store).get(handle.id)
    assert restored is not None
    assert restored.session.seats == ["human", "random"]
    assert restored.owned_seats([token]) == {0}


def test_eviction_drops_the_game_from_the_store(tmp_path: Path) -> None:
    store = GameStore(tmp_path)
    reg = GameRegistry(max_games=1, store=store)
    a = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
    # Idle well past the TTL (relative to monotonic(), which need not be large).
    a.touched = time.monotonic() - 100_000
    b = reg.create(GameSession(seed=1, n_players=2, seats=["human", "human"]))
    # A fresh store on the same db (as a restart would open) no longer has a.
    stored = {header["id"] for header, _ in GameStore(tmp_path).load()}
    assert a.id not in stored and b.id in stored


def test_restored_bot_game_resumes_playing(tmp_path: Path) -> None:
    store = GameStore(tmp_path)
    reg = GameRegistry(store=store)
    handle = reg.create(GameSession(seed=0, n_players=2, seats=["random", "random"]))
    for _ in range(5):  # a few bot moves, journalled (no driver runs here)
        if handle.session.bot_step() is not None:
            handle.bump()

    # A fresh app restores the position and its startup restarts the driver,
    # which plays the game out to the end.
    with TestClient(
        create_app(state_dir=str(tmp_path), bot_delay=0.0, warm=False)
    ) as c:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if c.get(f"/api/games/{handle.id}").json()["status"]["terminal"]:
                break
            time.sleep(0.1)
        body = c.get(f"/api/games/{handle.id}").json()
        assert body["status"]["terminal"] and body["status"]["winner"] is not None
