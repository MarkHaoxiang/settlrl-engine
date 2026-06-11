"""FastAPI endpoint tests for the renderer server.

Drive the API with a ``TestClient``: creating games, claiming seats (tokens),
per-seat snapshot views (your_turn / actions / hand redaction), applying a
legal move, rejecting an illegal one (409) or an unproven seat (403), and the
SPA 404-fallback that serves ``index.html`` for client-side routes. The SPA
tests are skipped if the built frontend (``frontend/dist``) is absent.
"""

import json
import threading

import pytest
from catan_render import server
from catan_render.games import GameRegistry
from catan_render.server import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    # Each test starts with an empty registry and no loaded replay, so tests
    # stay independent of execution order.
    server._GAMES = GameRegistry()
    server._REPLAY = None


def _create(**body: object) -> tuple[str, dict[str, str]]:
    """Create a game; return its id and the creator's {seat: token} claims."""
    resp = client.post("/api/games", json={"seed": 0, **body})
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    return doc["id"], dict(doc["tokens"])


def _hdr(tokens: dict[str, str]) -> dict[str, str]:
    return {"X-Seat-Tokens": ",".join(tokens.values())}


def test_create_claims_all_human_seats_by_default() -> None:
    game, tokens = _create()
    assert sorted(tokens) == ["0"]  # default seats: human + 3 random bots
    body = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()
    assert body["id"] == game
    assert body["status"]["your_turn"]
    assert len(body["actions"]) > 0
    assert body["seats_claimed"] == [0]


def test_unknown_game_404s() -> None:
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/action", json={"flat": 0}).status_code == 404


def test_spectator_view_is_redacted() -> None:
    game, _ = _create()
    body = client.get(f"/api/games/{game}").json()  # no tokens
    assert not body["status"]["your_turn"]
    assert body["actions"] == []
    assert body["belief"] is None
    assert all(p["resources"] is None for p in body["board"]["players"])
    assert all(p["dev_card_types"] is None for p in body["board"]["players"])
    # Public counts survive redaction.
    assert all("resource_cards" in p for p in body["board"]["players"])


def test_owned_seat_sees_own_hand_only() -> None:
    game, tokens = _create()
    body = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()
    players = {p["player"]: p for p in body["board"]["players"]}
    assert players[0]["resources"] is not None  # own seat
    assert all(players[p]["resources"] is None for p in (1, 2, 3))
    assert body["belief"] is not None and body["belief"]["observer"] == 0


def test_legal_action_requires_the_acting_seat() -> None:
    game, tokens = _create()
    flat = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"][0]["flat"]
    # Without tokens the move is refused before legality is even checked.
    assert client.post(f"/api/games/{game}/action", json={"flat": flat}).status_code == 403
    resp = client.post(f"/api/games/{game}/action", json={"flat": flat}, headers=_hdr(tokens))
    assert resp.status_code == 200
    assert "board" in resp.json()


def test_illegal_action_returns_409() -> None:
    game, tokens = _create()
    legal = {
        a["flat"]
        for a in client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"]
    }
    illegal = next(f for f in range(1000) if f not in legal)
    resp = client.post(f"/api/games/{game}/action", json={"flat": illegal}, headers=_hdr(tokens))
    assert resp.status_code == 409


def test_join_claims_remaining_human_seats() -> None:
    game, tokens = _create(seats=["human", "human", "random", "random"], claim="none")
    assert tokens == {}
    first = client.post(f"/api/games/{game}/join", json={}).json()
    assert first["seat"] == 0
    second = client.post(f"/api/games/{game}/join", json={"seat": 1}).json()
    assert second["seat"] == 1 and second["token"] != first["token"]
    # All human seats claimed now.
    assert client.post(f"/api/games/{game}/join", json={}).status_code == 409
    # A bot seat can never be claimed.
    assert client.post(f"/api/games/{game}/join", json={"seat": 2}).status_code == 409


def test_two_humans_see_their_own_turns() -> None:
    game, _ = _create(seats=["human", "human", "random", "random"], claim="none")
    a = client.post(f"/api/games/{game}/join", json={"seat": 0}).json()
    b = client.post(f"/api/games/{game}/join", json={"seat": 1}).json()
    ha = {"X-Seat-Tokens": a["token"]}
    hb = {"X-Seat-Tokens": b["token"]}
    # Seat 0 acts first in setup; seat 1 must wait (and can't act for them).
    view_a = client.get(f"/api/games/{game}", headers=ha).json()
    assert view_a["status"]["your_turn"]
    view_b = client.get(f"/api/games/{game}", headers=hb).json()
    assert not view_b["status"]["your_turn"] and view_b["actions"] == []
    flat = view_a["actions"][0]["flat"]
    assert (
        client.post(f"/api/games/{game}/action", json={"flat": flat}, headers=hb).status_code
        == 403
    )


def test_create_two_players_and_spiral_is_deterministic() -> None:
    game, tokens = _create(n_players=2, number_placement="spiral", seed=7)
    body = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()
    assert len(body["board"]["players"]) == 2
    # Same seed + placement reproduces the same board in a fresh game.
    again, _ = _create(n_players=2, number_placement="spiral", seed=7)
    body2 = client.get(f"/api/games/{again}").json()
    assert body2["board"]["tiles"] == body["board"]["tiles"]


def test_create_rejects_unsupported_player_counts() -> None:
    for bad in (1, 3, 5):
        resp = client.post("/api/games", json={"seed": 0, "n_players": bad})
        assert resp.status_code == 422


def test_create_rejects_bad_seats() -> None:
    resp = client.post(
        "/api/games", json={"seed": 0, "seats": ["human", "clever", "random", "random"]}
    )
    assert resp.status_code == 422
    resp = client.post("/api/games", json={"seed": 0, "seats": ["human", "random"]})
    assert resp.status_code == 422


def test_bot_endpoint_steps_one_move_and_reports_it() -> None:
    game, _ = _create(seats=["random"] * 4)
    body = client.post(f"/api/games/{game}/bot").json()
    assert body["bot_move"] is not None
    assert body["bot_move"]["player"] == 0
    assert body["bot_move"]["action"]["type"] == "setup_settlement"


def test_all_bot_game_is_spectated() -> None:
    game, tokens = _create(seats=["random"] * 4)
    assert tokens == {}
    body = client.get(f"/api/games/{game}").json()
    assert not body["status"]["your_turn"]
    assert body["belief"] is None


def test_concurrent_duplicate_actions_apply_once() -> None:
    game, tokens = _create()
    flat = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"][0]["flat"]
    codes: list[int] = []

    def post() -> None:
        codes.append(
            client.post(
                f"/api/games/{game}/action", json={"flat": flat}, headers=_hdr(tokens)
            ).status_code
        )

    threads = [threading.Thread(target=post) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # The per-game lock serialises: exactly one apply succeeds, the rest see
    # the move as no-longer-legal (409).
    assert sorted(codes) == [200, 409, 409, 409]


def test_moves_are_logged() -> None:
    game, tokens = _create()
    flat = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"][0]["flat"]
    body = client.post(
        f"/api/games/{game}/action", json={"flat": flat}, headers=_hdr(tokens)
    ).json()
    moves = [e for e in body["log"] if e["kind"] == "move"]
    assert moves and moves[-1]["player"] == 0


def test_chat_requires_seat_ownership() -> None:
    game, tokens = _create()
    body = client.post(
        f"/api/games/{game}/chat", json={"text": "hi", "player": 0}, headers=_hdr(tokens)
    ).json()
    assert body["log"][-1]["kind"] == "chat" and body["log"][-1]["player"] == 0
    # Unowned seat: refused. Spectator (no seat claimed): allowed.
    assert (
        client.post(f"/api/games/{game}/chat", json={"text": "hi", "player": 1}).status_code
        == 403
    )
    body = client.post(f"/api/games/{game}/chat", json={"text": "gl"}).json()
    assert body["log"][-1]["player"] is None


def test_chat_rejects_blank_text() -> None:
    game, _ = _create()
    assert client.post(f"/api/games/{game}/chat", json={"text": "   "}).status_code == 422


def _finish(game: str) -> None:
    """Drive an all-bot game to completion in-process (HTTP would be slow)."""
    handle = server._GAMES.get(game)
    assert handle is not None
    handle.session._run_bots()
    assert handle.session.terminal()


def test_record_refused_while_running_then_exports() -> None:
    game, _ = _create(seats=["random"] * 4)
    # A live game's record would reconstruct hidden hands when replayed.
    assert client.get(f"/api/games/{game}/record").status_code == 409
    _finish(game)
    resp = client.get(f"/api/games/{game}/record")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["winner"] is not None and len(doc["moves"]) > 0


def test_replay_from_game_and_scrub() -> None:
    game, _ = _create(seats=["random"] * 4)
    assert client.post(f"/api/games/{game}/replay").status_code == 409  # running
    _finish(game)
    opening = client.post(f"/api/games/{game}/replay").json()
    assert opening["move"] == 0 and opening["n_moves"] > 0
    mid = client.get("/api/replay/state", params={"move": 5}).json()
    assert mid["move"] == 5
    last = client.get("/api/replay/state", params={"move": opening["n_moves"]}).json()
    assert last["winner"] is not None
    assert client.get("/api/replay/state", params={"move": 99999}).status_code == 422


def test_replay_upload_roundtrip() -> None:
    game, _ = _create(seats=["random"] * 4)
    _finish(game)
    handle = server._GAMES.get(game)
    assert handle is not None
    doc = json.loads(handle.session.record().to_json())
    assert client.post("/api/replay", json=doc).status_code == 200
    assert client.get("/api/replay/record").status_code == 200


def test_replay_state_404_until_loaded() -> None:
    assert client.get("/api/replay/state").status_code == 404


def test_replay_rejects_bad_records() -> None:
    assert client.post("/api/replay", json={"seed": 1}).status_code == 422
    assert (
        client.post(
            "/api/replay",
            json={
                "seed": 1,
                "n_players": 4,
                "number_placement": "random",
                "moves": [{"player": 0, "flat": 9}],
                "winner": None,
            },
        ).status_code
        == 422
    )


def test_get_bots_lists_policies() -> None:
    resp = client.get("/api/bots")
    assert resp.status_code == 200
    body = resp.json()
    assert "random" in body
    assert all("counts" in spec and "params" in spec for spec in body.values())


_DIST = server._dist.exists()


@pytest.mark.skipif(not _DIST, reason="frontend/dist not built")
def test_spa_fallback_serves_index_for_client_route() -> None:
    resp = client.get("/play")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.skipif(not _DIST, reason="frontend/dist not built")
def test_spa_fallback_404_for_missing_asset() -> None:
    assert client.get("/assets/nope.js").status_code == 404
