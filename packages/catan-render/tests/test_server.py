"""FastAPI endpoint tests for the renderer server.

Drive the API with a ``TestClient``: the board / game snapshots, applying a legal
move, rejecting an illegal one (409), resetting, and the SPA 404-fallback that
serves ``index.html`` for client-side routes. The SPA tests are skipped if the
built frontend (``frontend/dist``) is absent.
"""

import pytest
from fastapi.testclient import TestClient

from catan_render import server
from catan_render.server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_game() -> None:
    # Each test starts from a deterministic fresh game (server holds one global
    # session); reset keeps tests independent of execution order (including the
    # seat count a previous test may have changed).
    server._SESSION.reset(0, n_players=4)


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


def test_reset_rejects_unsupported_player_counts() -> None:
    # The renderer offers 2 and 4 seats for now (422 from request validation).
    for bad in (1, 3, 5):
        resp = client.post("/api/game/reset", json={"seed": 0, "n_players": bad})
        assert resp.status_code == 422


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
