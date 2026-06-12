"""Persistence: journals on disk replay back into live games on restart."""

import time
from pathlib import Path

from catan_render.games import GameHandle, GameRegistry, restore_registry
from catan_render.server import create_app
from catan_render.session import GameSession, GameSetup
from catan_render.store import GameStore
from fastapi.testclient import TestClient


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


def _play(handle: GameHandle, n: int) -> None:
    """Apply ``n`` legal moves to a handle, journalling each (as a route would)."""
    for _ in range(n):
        with handle.lock:
            flat = int(handle.session.legal_flat()[0])
            handle.session.apply(flat)
            handle.bump()


def test_restart_resumes_an_in_progress_game(tmp_path: Path) -> None:
    # First boot: create a game, play a move, leave a chat line.
    with TestClient(create_app(state_dir=str(tmp_path))) as c1:
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
    with TestClient(create_app(state_dir=str(tmp_path))) as c2:
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


def test_eviction_deletes_the_journal(tmp_path: Path) -> None:
    store = GameStore(tmp_path)
    reg = GameRegistry(max_games=1, store=store)
    a = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
    path = tmp_path / f"{a.id}.jsonl"
    assert path.exists()
    a.touched = 0.0  # unstarted and idle -> evictable
    b = reg.create(GameSession(seed=1, n_players=2, seats=["human", "human"]))
    assert not path.exists()
    assert (tmp_path / f"{b.id}.jsonl").exists()


def test_torn_final_write_is_dropped_on_load(tmp_path: Path) -> None:
    store = GameStore(tmp_path)
    reg = GameRegistry(store=store)
    handle = reg.create(GameSession(seed=0, n_players=2, seats=["human", "human"]))
    _play(handle, 1)
    handle.journal.close()  # type: ignore[union-attr]
    # A crash mid-append leaves a partial last line.
    path = tmp_path / f"{handle.id}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"t":"move","fl')
    restored = restore_registry(store).get(handle.id)
    assert restored is not None
    assert restored.session.moves_played == 1  # the good move kept, the torn one gone


def test_restored_bot_game_resumes_playing(tmp_path: Path) -> None:
    store = GameStore(tmp_path)
    reg = GameRegistry(store=store)
    handle = reg.create(GameSession(seed=0, n_players=2, seats=["random", "random"]))
    for _ in range(5):  # a few bot moves, journalled
        with handle.lock:
            if handle.session.bot_step() is not None:
                handle.bump()
    handle.journal.close()  # type: ignore[union-attr]

    # A fresh app restores the position and its startup restarts the driver,
    # which plays the game out to the end.
    with TestClient(create_app(state_dir=str(tmp_path), bot_delay=0.0)) as c:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if c.get(f"/api/games/{handle.id}").json()["status"]["terminal"]:
                break
            time.sleep(0.1)
        body = c.get(f"/api/games/{handle.id}").json()
        assert body["status"]["terminal"] and body["status"]["winner"] is not None
