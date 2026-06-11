"""Route-level tests for the FastAPI server.

Each test builds its own app around its own registry (``create_app``), so
nothing is shared between tests. These cover what the routes themselves own —
auth and status codes, locking, request plumbing — plus the SPA fallback; the
per-seat view contents live in ``test_views.py`` and registry logic in
``test_games.py``.
"""

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from catan_render.games import GameRegistry
from catan_render.server import create_app
from fastapi.testclient import TestClient


@pytest.fixture()
def registry() -> GameRegistry:
    return GameRegistry()


@pytest.fixture()
def client(registry: GameRegistry) -> Iterator[TestClient]:
    yield TestClient(create_app(registry))


def _create(client: TestClient, **body: object) -> tuple[str, dict[str, str]]:
    """Create a game; return its id and the creator's {seat: token} claims."""
    resp = client.post("/api/games", json={"seed": 0, **body})
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    return doc["id"], dict(doc["tokens"])


def _hdr(tokens: dict[str, str]) -> dict[str, str]:
    return {"X-Seat-Tokens": ",".join(tokens.values())}


def test_create_claims_all_human_seats_by_default(client: TestClient) -> None:
    game, tokens = _create(client)
    assert sorted(tokens) == ["0"]  # default seats: human + 3 random bots
    body = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()
    assert body["id"] == game
    assert body["status"]["your_turn"] and len(body["actions"]) > 0
    assert body["seats_claimed"] == [0]


def test_unknown_game_404s(client: TestClient) -> None:
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/action", json={"flat": 0}).status_code == 404


def test_action_requires_the_acting_seats_token(client: TestClient) -> None:
    game, tokens = _create(client, seats=["human", "human", "random", "random"], claim="none")
    a = client.post(f"/api/games/{game}/join", json={"seat": 0}).json()
    b = client.post(f"/api/games/{game}/join", json={"seat": 1}).json()
    flat = client.get(f"/api/games/{game}", headers={"X-Seat-Tokens": a["token"]}).json()[
        "actions"
    ][0]["flat"]
    # No token, and the wrong seat's token: refused before legality.
    assert client.post(f"/api/games/{game}/action", json={"flat": flat}).status_code == 403
    assert (
        client.post(
            f"/api/games/{game}/action", json={"flat": flat}, headers={"X-Seat-Tokens": b["token"]}
        ).status_code
        == 403
    )
    resp = client.post(
        f"/api/games/{game}/action", json={"flat": flat}, headers={"X-Seat-Tokens": a["token"]}
    )
    assert resp.status_code == 200


def test_illegal_action_returns_409(client: TestClient) -> None:
    game, tokens = _create(client)
    legal = {
        a["flat"]
        for a in client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"]
    }
    illegal = next(f for f in range(1000) if f not in legal)
    resp = client.post(f"/api/games/{game}/action", json={"flat": illegal}, headers=_hdr(tokens))
    assert resp.status_code == 409


def test_join_conflicts_are_409(client: TestClient) -> None:
    game, tokens = _create(client, seats=["human", "human", "random", "random"], claim="none")
    assert tokens == {}
    assert client.post(f"/api/games/{game}/join", json={}).json()["seat"] == 0
    assert client.post(f"/api/games/{game}/join", json={"seat": 1}).status_code == 200
    assert client.post(f"/api/games/{game}/join", json={}).status_code == 409  # full
    assert client.post(f"/api/games/{game}/join", json={"seat": 2}).status_code == 409  # bot


def test_create_rejects_bad_requests(client: TestClient) -> None:
    for bad in (1, 3, 5):
        assert client.post("/api/games", json={"seed": 0, "n_players": bad}).status_code == 422
    assert (
        client.post(
            "/api/games", json={"seed": 0, "seats": ["human", "clever", "random", "random"]}
        ).status_code
        == 422
    )
    assert client.post("/api/games", json={"seed": 0, "seats": ["human", "random"]}).status_code == 422


def test_create_same_seed_reproduces_the_board(client: TestClient) -> None:
    game, _ = _create(client, n_players=2, number_placement="spiral", seed=7)
    again, _ = _create(client, n_players=2, number_placement="spiral", seed=7)
    a = client.get(f"/api/games/{game}").json()["board"]["tiles"]
    b = client.get(f"/api/games/{again}").json()["board"]["tiles"]
    assert a == b


def test_bot_endpoint_steps_one_move_and_reports_it(client: TestClient) -> None:
    game, _ = _create(client, seats=["random"] * 4)
    body = client.post(f"/api/games/{game}/bot").json()
    assert body["bot_move"] is not None
    assert body["bot_move"]["player"] == 0
    assert body["bot_move"]["action"]["type"] == "setup_settlement"


def test_concurrent_duplicate_actions_apply_once(client: TestClient) -> None:
    game, tokens = _create(client)
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


def test_chat_requires_seat_ownership(client: TestClient) -> None:
    game, tokens = _create(client)
    body = client.post(
        f"/api/games/{game}/chat", json={"text": "hi", "player": 0}, headers=_hdr(tokens)
    ).json()
    assert body["log"][-1]["kind"] == "chat" and body["log"][-1]["player"] == 0
    # Unowned seat: refused. Spectator (no seat given): allowed.
    assert (
        client.post(f"/api/games/{game}/chat", json={"text": "hi", "player": 1}).status_code == 403
    )
    assert client.post(f"/api/games/{game}/chat", json={"text": "gl"}).json()["log"][-1]["player"] is None
    assert client.post(f"/api/games/{game}/chat", json={"text": "   "}).status_code == 422


def _finish(registry: GameRegistry, game: str) -> None:
    """Drive an all-bot game to completion in-process (HTTP would be slow)."""
    handle = registry.get(game)
    assert handle is not None
    handle.session._run_bots()
    assert handle.session.terminal()


def test_record_and_replay_export_finished_games_only(
    client: TestClient, registry: GameRegistry
) -> None:
    game, _ = _create(client, seats=["random"] * 4)
    # A live game's record would reconstruct hidden hands when replayed.
    assert client.get(f"/api/games/{game}/record").status_code == 409
    assert client.post(f"/api/games/{game}/replay").status_code == 409
    _finish(registry, game)
    doc = client.get(f"/api/games/{game}/record").json()
    assert doc["winner"] is not None and len(doc["moves"]) > 0
    opening = client.post(f"/api/games/{game}/replay").json()
    assert opening["move"] == 0 and opening["n_moves"] == len(doc["moves"])
    mid = client.get("/api/replay/state", params={"move": 5}).json()
    assert mid["move"] == 5
    assert client.get("/api/replay/state", params={"move": 99999}).status_code == 422


def test_replay_upload_roundtrip(client: TestClient, registry: GameRegistry) -> None:
    game, _ = _create(client, seats=["random"] * 4)
    _finish(registry, game)
    doc = client.get(f"/api/games/{game}/record").json()
    assert client.post("/api/replay", json=doc).status_code == 200
    assert client.get("/api/replay/record").status_code == 200


def test_replay_state_404_until_loaded(client: TestClient) -> None:
    assert client.get("/api/replay/state").status_code == 404


def test_replay_rejects_bad_records(client: TestClient) -> None:
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


def test_get_bots_lists_policies(client: TestClient) -> None:
    body = client.get("/api/bots").json()
    assert "random" in body
    assert all("counts" in spec and "params" in spec for spec in body.values())


def test_openapi_schema_is_committed() -> None:
    # The committed schema generates the frontend's wire types; regen with
    # `npm run gen-api` in frontend/ whenever the models change.
    committed = json.loads(
        (Path(__file__).parent.parent / "frontend" / "openapi.json").read_text()
    )
    assert create_app().openapi() == committed, "schema drift: run `npm run gen-api`"


_DIST = (Path(__file__).parent.parent / "frontend" / "dist").exists()


@pytest.mark.skipif(not _DIST, reason="frontend/dist not built")
def test_spa_fallback_serves_index_for_client_route(client: TestClient) -> None:
    resp = client.get("/play")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.skipif(not _DIST, reason="frontend/dist not built")
def test_spa_fallback_404_for_missing_asset(client: TestClient) -> None:
    assert client.get("/assets/nope.js").status_code == 404
