"""FastAPI endpoint tests for the renderer server.

Drive the API with a ``TestClient``: the board / game snapshots, applying a legal
move, rejecting an illegal one (409), resetting, and the SPA 404-fallback that
serves ``index.html`` for client-side routes. The SPA tests are skipped if the
built frontend (``frontend/dist``) is absent.
"""

import threading

import pytest
from catan_render import server
from catan_render.server import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_game() -> None:
    # Each test starts from a deterministic fresh game (server holds one global
    # session); reset keeps tests independent of execution order (including the
    # seat count a previous test may have changed). Any loaded replay is
    # dropped too.
    server._SESSION.reset(0, n_players=4)
    server._REPLAY = None


def test_get_board() -> None:
    resp = client.get("/api/board")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["tiles"]) == 19
    assert len(body["ports"]) == 9
    assert len(body["players"]) == 4


def test_get_game() -> None:
    resp = client.get("/api/game")
    assert resp.status_code == 200
    body = resp.json()
    assert "board" in body and "status" in body and "actions" in body
    # Opening setup phase: it's the human's turn with legal moves on offer.
    assert body["status"]["your_turn"]
    assert len(body["actions"]) > 0


def test_legal_action() -> None:
    game = client.get("/api/game").json()
    flat = game["actions"][0]["flat"]
    resp = client.post("/api/game/action", json={"flat": flat})
    assert resp.status_code == 200
    assert "board" in resp.json()


def test_illegal_action_returns_409() -> None:
    game = client.get("/api/game").json()
    legal = {a["flat"] for a in game["actions"]}
    illegal = next(f for f in range(1000) if f not in legal)
    resp = client.post("/api/game/action", json={"flat": illegal})
    assert resp.status_code == 409


def test_reset() -> None:
    resp = client.post("/api/game/reset", json={"seed": 7})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"]["your_turn"]


def test_reset_two_players() -> None:
    resp = client.post("/api/game/reset", json={"seed": 7, "n_players": 2})
    assert resp.status_code == 200
    body = resp.json()
    # Only the two seated players get a panel; play starts as usual.
    assert len(body["board"]["players"]) == 2
    assert body["status"]["your_turn"]
    # The seat count sticks across subsequent moves.
    flat = body["actions"][0]["flat"]
    body = client.post("/api/game/action", json={"flat": flat}).json()
    assert len(body["board"]["players"]) == 2


def test_reset_spiral_numbers() -> None:
    resp = client.post(
        "/api/game/reset", json={"seed": 7, "number_placement": "spiral"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"]["your_turn"]
    # Same seed + placement reproduces the same board.
    tiles = resp.json()["board"]["tiles"]
    again = client.post(
        "/api/game/reset", json={"seed": 7, "number_placement": "spiral"}
    )
    assert again.json()["board"]["tiles"] == tiles


def test_reset_rejects_unsupported_player_counts() -> None:
    # The renderer offers 2 and 4 seats for now (422 from request validation).
    for bad in (1, 3, 5):
        resp = client.post("/api/game/reset", json={"seed": 0, "n_players": bad})
        assert resp.status_code == 422


def test_bot_endpoint_steps_one_move_and_reports_it() -> None:
    # The human plays both setup placements; then the bot seats are due.
    for _ in range(2):
        game = client.get("/api/game").json()
        client.post("/api/game/action", json={"flat": game["actions"][0]["flat"]})
    body = client.post("/api/game/bot").json()
    assert body["bot_move"] is not None
    assert body["bot_move"]["player"] == 1
    assert body["bot_move"]["action"]["label"]
    # Stepping repeatedly hands the turn back to the human...
    for _ in range(100):
        if body["status"]["your_turn"]:
            break
        body = client.post("/api/game/bot").json()
    assert body["status"]["your_turn"]
    # ...after which no bot move is due.
    assert client.post("/api/game/bot").json()["bot_move"] is None


def test_get_bots_lists_policies() -> None:
    resp = client.get("/api/bots")
    assert resp.status_code == 200
    bots = resp.json()
    assert "random" in bots and "human" not in bots
    # Each kind maps to the player counts it supports.
    assert set(bots["random"]) >= {2, 4}


def test_reset_rejects_seat_kind_unsupported_at_count() -> None:
    bots = client.get("/api/bots").json()
    two_only = [k for k, counts in bots.items() if 2 in counts and 4 not in counts]
    if not two_only:
        pytest.skip("no two-player-only bot kinds")
    resp = client.post(
        "/api/game/reset",
        json={
            "seed": 0,
            "n_players": 4,
            "seats": ["human", two_only[0], "random", "random"],
        },
    )
    assert resp.status_code == 422


def test_reset_with_seats() -> None:
    resp = client.post(
        "/api/game/reset",
        json={"seed": 7, "n_players": 2, "seats": ["human", "random"]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"]["seats"] == ["human", "random"]


def test_reset_all_bot_seats_spectates() -> None:
    # No human seat: never your_turn, and the bot endpoint plays from move one
    # (seat 0, a bot, opens the game).
    resp = client.post(
        "/api/game/reset",
        json={"seed": 7, "n_players": 2, "seats": ["random", "random"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert not body["status"]["your_turn"]
    assert body["actions"] == []
    body = client.post("/api/game/bot").json()
    assert body["bot_move"] is not None
    assert body["bot_move"]["player"] == 0


def test_reset_rejects_bad_seats() -> None:
    # Unknown bot kind, and a list whose length doesn't match the seat count.
    for bad in (["human", "clever"], ["human", "random", "random"]):
        resp = client.post(
            "/api/game/reset", json={"seed": 0, "n_players": 2, "seats": bad}
        )
        assert resp.status_code == 422


def test_concurrent_duplicate_actions_apply_once() -> None:
    # FastAPI handles requests in a threadpool: a double-clicked move arriving
    # twice concurrently must apply once (the loser gets the usual 409).
    flat = client.get("/api/game").json()["actions"][0]["flat"]
    results: list[int] = []
    post = lambda: results.append(  # noqa: E731
        client.post("/api/game/action", json={"flat": flat}).status_code
    )
    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [200, 409]
    assert len(client.get("/api/game").json()["log"]) == 1


def test_moves_are_logged() -> None:
    game = client.get("/api/game").json()
    assert game["log"] == []
    flat = game["actions"][0]["flat"]
    body = client.post("/api/game/action", json={"flat": flat}).json()
    (entry,) = body["log"]
    assert entry["kind"] == "move" and entry["player"] == 0
    assert entry["action_type"] == "setup_settlement"


def test_chat_endpoint_appends_to_log() -> None:
    resp = client.post("/api/game/chat", json={"text": "  hi there  ", "player": 0})
    assert resp.status_code == 200
    entry = resp.json()["log"][-1]
    assert entry["kind"] == "chat"
    assert entry["player"] == 0
    assert entry["text"] == "hi there"
    # A reset starts a fresh log.
    assert client.post("/api/game/reset", json={"seed": 1}).json()["log"] == []


def test_record_endpoint_exports_the_game() -> None:
    game = client.get("/api/game").json()
    client.post("/api/game/action", json={"flat": game["actions"][0]["flat"]})
    resp = client.get("/api/game/record")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["version"] == 1
    assert doc["n_players"] == 4 and doc["winner"] is None
    assert doc["meta"]["seats"] == ["human", "random", "random", "random"]
    (move,) = doc["moves"]
    assert move["player"] == 0 and move["type"] == "setup_settlement"


def test_chat_rejects_blank_text_and_bad_seat() -> None:
    assert client.post("/api/game/chat", json={"text": "   "}).status_code == 422
    assert (
        client.post("/api/game/chat", json={"text": "hi", "player": 9}).status_code
        == 422
    )


def test_replay_endpoints_load_and_scrub() -> None:
    # Play two human moves, load the live game as a replay, and scrub it.
    for _ in range(2):
        game = client.get("/api/game").json()
        client.post("/api/game/action", json={"flat": game["actions"][0]["flat"]})
    body = client.post("/api/replay/from-game").json()
    assert body["move"] == 0 and body["n_moves"] == 2
    assert body["log"] == []  # nothing played yet at the opening board
    assert body["board"]["buildings"] == []
    # After both moves the settlement + road are on the board and in the log.
    end = client.get("/api/replay/state", params={"move": 2}).json()
    assert len(end["board"]["buildings"]) == 1 and len(end["board"]["roads"]) == 1
    assert [e["action_type"] for e in end["log"]] == ["setup_settlement", "setup_road"]
    # Out-of-range moves are rejected.
    assert client.get("/api/replay/state", params={"move": 3}).status_code == 422
    assert client.get("/api/replay/state", params={"move": -1}).status_code == 422


def test_replay_upload_roundtrip() -> None:
    game = client.get("/api/game").json()
    client.post("/api/game/action", json={"flat": game["actions"][0]["flat"]})
    record = client.get("/api/game/record").json()
    body = client.post("/api/replay", json=record).json()
    assert body["n_moves"] == 1
    assert body["seats"] == ["human", "random", "random", "random"]
    # The loaded record can be downloaded back unchanged.
    assert client.get("/api/replay/record").json() == record


def test_replay_state_404_until_loaded() -> None:
    assert client.get("/api/replay/state").status_code == 404
    assert client.get("/api/replay/record").status_code == 404


def test_replay_rejects_bad_records() -> None:
    assert client.post("/api/replay", json={"version": 99}).status_code == 422
    # A structurally valid record whose moves don't replay (illegal move 0).
    bad = {
        "version": 1,
        "seed": 0,
        "n_players": 4,
        "number_placement": "random",
        "winner": None,
        "moves": [{"player": 1, "flat": 0}],
    }
    assert client.post("/api/replay", json=bad).status_code == 422


def test_replay_of_finished_game_reports_winner() -> None:
    # An all-bot game plays itself out; its replay carries the winner and the
    # win line appears only at the final move.
    client.post(
        "/api/game/reset",
        json={"seed": 7, "n_players": 2, "seats": ["random", "random"]},
    )
    for _ in range(50_000):
        if client.post("/api/game/bot").json()["bot_move"] is None:
            break
    body = client.post("/api/replay/from-game").json()
    n, winner = body["n_moves"], body["winner"]
    assert winner is not None
    end = client.get("/api/replay/state", params={"move": n}).json()
    assert end["log"][-1]["kind"] == "win" and end["log"][-1]["player"] == winner
    before = client.get("/api/replay/state", params={"move": n - 1}).json()
    assert all(e["kind"] != "win" for e in before["log"])


_HAS_DIST = (server._dist / "index.html").exists()
spa = pytest.mark.skipif(not _HAS_DIST, reason="frontend/dist not built")


@spa
def test_spa_fallback_serves_index_for_client_route() -> None:
    # An extension-less unknown path (a client-side route) falls back to the SPA.
    resp = client.get("/play")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@spa
def test_spa_fallback_404_for_missing_asset() -> None:
    # A path that looks like a file (has an extension) still 404s when missing.
    resp = client.get("/assets/does-not-exist.js")
    assert resp.status_code == 404
