"""Route-level tests for the FastAPI server.

Each test builds its own app around its own registry (``create_app``), so
nothing is shared between tests. These cover what the routes themselves own —
auth and status codes, locking, request plumbing — plus the SPA fallback; the
per-seat view contents live in ``test_views.py`` and registry logic in
``test_games.py``.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from _helpers import bot_registry
from fastapi import FastAPI
from fastapi.testclient import TestClient
from settlrl_app.game.games import GameRegistry
from settlrl_app.server import create_app


@pytest.fixture()
def registry() -> GameRegistry:
    return GameRegistry()


@pytest.fixture()
def client(registry: GameRegistry) -> Iterator[TestClient]:
    # The ``with`` form runs the lifespan and keeps one event loop alive for the
    # client's lifetime, so background driver tasks and the per-game asyncio
    # locks share a single loop across requests. Bot-backed so the bot seats
    # these tests use validate and play (the server runs no bots itself).
    with TestClient(create_app(registry, providers=bot_registry())) as client:
        yield client


def _create(client: TestClient, **body: object) -> tuple[str, dict[str, str]]:
    """Create a game; return its id and the creator's {seat: token} claims."""
    resp = client.post("/api/games", json={"seed": 0, **body})
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    return doc["id"], dict(doc["tokens"])


def _hdr(tokens: dict[str, str]) -> dict[str, str]:
    return {"X-Seat-Tokens": ",".join(tokens.values())}


def test_two_player_games_play_to_fifteen_others_to_ten(client: TestClient) -> None:
    game2, t2 = _create(client, n_players=2, seats=["human", "human"])
    st2 = client.get(f"/api/games/{game2}", headers=_hdr(t2)).json()["status"]
    assert st2["victory_points_to_win"] == 15

    game4, t4 = _create(client, n_players=4)
    st4 = client.get(f"/api/games/{game4}", headers=_hdr(t4)).json()["status"]
    assert st4["victory_points_to_win"] == 10


def test_create_claims_all_human_seats_by_default(client: TestClient) -> None:
    # One human seat among bots: claim="all" (the default) claims just it.
    game, tokens = _create(client, seats=["human", "random", "random", "random"])
    assert sorted(tokens) == ["0"]
    body = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()
    assert body["id"] == game
    assert body["status"]["your_turn"] and len(body["actions"]) > 0
    assert body["seats_claimed"] == [0]


def test_create_claim_first_takes_one_seat_and_leaves_the_rest(
    client: TestClient,
) -> None:
    game, tokens = _create(
        client, seats=["human", "human", "random", "random"], claim="first"
    )
    assert sorted(tokens) == ["0"]
    assert client.post(f"/api/games/{game}/join", json={}).json()["seat"] == 1


def test_full_registry_of_active_games_returns_503() -> None:
    with TestClient(create_app(GameRegistry(max_games=1))) as client:
        assert client.post("/api/games", json={"seed": 0}).status_code == 200
        assert client.post("/api/games", json={"seed": 0}).status_code == 503


def test_create_queues_past_the_active_cap() -> None:
    # max_active=1: the second creator gets a 202 with their place in line.
    with TestClient(create_app(GameRegistry(max_games=8, max_active=1))) as client:
        body = {"seed": 0, "n_players": 2, "seats": ["human", "human"]}
        assert client.post("/api/games", json=body).status_code == 200
        queued = client.post("/api/games", json=body)
        assert queued.status_code == 202
        doc = queued.json()
        assert doc["queued"] is True and doc["ticket"]
        assert doc["position"] == 1 and doc["total"] == 1


def test_unknown_game_404s(client: TestClient) -> None:
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/action", json={"flat": 0}).status_code == 404


def test_action_requires_the_acting_seats_token(client: TestClient) -> None:
    game, _ = _create(
        client, seats=["human", "human", "random", "random"], claim="none"
    )
    a = client.post(f"/api/games/{game}/join", json={"seat": 0}).json()
    b = client.post(f"/api/games/{game}/join", json={"seat": 1}).json()
    flat = client.get(
        f"/api/games/{game}", headers={"X-Seat-Tokens": a["token"]}
    ).json()["actions"][0]["flat"]
    # No token, and the wrong seat's token: refused before legality.
    assert (
        client.post(f"/api/games/{game}/action", json={"flat": flat}).status_code == 403
    )
    assert (
        client.post(
            f"/api/games/{game}/action",
            json={"flat": flat},
            headers={"X-Seat-Tokens": b["token"]},
        ).status_code
        == 403
    )
    resp = client.post(
        f"/api/games/{game}/action",
        json={"flat": flat},
        headers={"X-Seat-Tokens": a["token"]},
    )
    assert resp.status_code == 200


def test_illegal_action_returns_409(client: TestClient) -> None:
    game, tokens = _create(client)
    legal = {
        a["flat"]
        for a in client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()[
            "actions"
        ]
    }
    illegal = next(f for f in range(1000) if f not in legal)
    resp = client.post(
        f"/api/games/{game}/action", json={"flat": illegal}, headers=_hdr(tokens)
    )
    assert resp.status_code == 409


def test_join_conflicts_are_409(client: TestClient) -> None:
    game, tokens = _create(
        client, seats=["human", "human", "random", "random"], claim="none"
    )
    assert tokens == {}
    assert client.post(f"/api/games/{game}/join", json={}).json()["seat"] == 0
    assert client.post(f"/api/games/{game}/join", json={"seat": 1}).status_code == 200
    assert client.post(f"/api/games/{game}/join", json={}).status_code == 409  # full
    assert (
        client.post(f"/api/games/{game}/join", json={"seat": 2}).status_code == 409
    )  # bot


def test_create_rejects_bad_requests(client: TestClient) -> None:
    for bad in (1, 3, 5):
        assert (
            client.post("/api/games", json={"seed": 0, "n_players": bad}).status_code
            == 422
        )
    assert (
        client.post(
            "/api/games",
            json={"seed": 0, "seats": ["human", "clever", "random", "random"]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/games", json={"seed": 0, "seats": ["human", "random"]}
        ).status_code
        == 422
    )


def test_create_same_seed_reproduces_the_board(client: TestClient) -> None:
    game, _ = _create(client, n_players=2, number_placement="spiral", seed=7)
    again, _ = _create(client, n_players=2, number_placement="spiral", seed=7)
    a = client.get(f"/api/games/{game}").json()["board"]["tiles"]
    b = client.get(f"/api/games/{again}").json()["board"]["tiles"]
    assert a == b


def test_preview_returns_a_board_without_creating_a_game(client: TestClient) -> None:
    # The map picker: a board for the seed, no game created.
    spiral = client.get(
        "/api/preview", params={"seed": 5, "number_placement": "spiral"}
    ).json()
    assert len(spiral["tiles"]) == 19 and len(spiral["ports"]) == 9 and spiral["robber"]
    # Same seed, the other placement: same terrain, the numbers move.
    rand = client.get(
        "/api/preview", params={"seed": 5, "number_placement": "random"}
    ).json()
    terrain = [t["terrain"] for t in spiral["tiles"]]
    assert terrain == [t["terrain"] for t in rand["tiles"]]
    assert [t["number"] for t in spiral["tiles"]] != [
        t["number"] for t in rand["tiles"]
    ]
    assert client.get("/api/preview", params={"n_players": 5}).status_code == 422


@contextmanager
def _live_server(app: FastAPI) -> Iterator[int]:
    """Run ``app`` on a real uvicorn server (ephemeral port), yielding the port.

    SSE streams need this: TestClient buffers whole responses and would block
    on an open stream forever.
    """
    # A short graceful-shutdown timeout: an SSE generator parked in its keepalive
    # wait() would otherwise hold teardown for the full keepalive interval.
    config = uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=1)
    server = uvicorn.Server(config)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, args=([sock],), daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.02)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _next_event(lines: Iterator[str]) -> dict[str, object]:
    """The next SSE data event (skipping keepalives), parsed."""
    for _, line in zip(range(50), lines, strict=False):
        if line.startswith("data:"):
            return dict(json.loads(line[5:].strip()))
    raise AssertionError("no data event arrived")


def test_events_stream_snapshot_now_then_on_every_change() -> None:
    # The never-ending stream needs a real server: TestClient buffers whole
    # responses, so it would block on the open stream forever.
    with (
        _live_server(create_app(GameRegistry())) as port,
        httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as http,
    ):
        # All-human so no bot driver mutates the game mid-test.
        doc = http.post("/api/games", json={"seed": 0, "seats": ["human"] * 4}).json()
        game, hdr = doc["id"], _hdr(dict(doc["tokens"]))
        with http.stream("GET", f"/api/games/{game}/events", headers=hdr) as resp:
            lines = resp.iter_lines()
            first = _next_event(lines)
            assert first["status"]["your_turn"] is True  # type: ignore[index]
            flat = first["actions"][0]["flat"]  # type: ignore[index]
            http.post(f"/api/games/{game}/action", json={"flat": flat}, headers=hdr)
            second = _next_event(lines)
    assert second["version"] > first["version"]  # type: ignore[operator]
    assert len(second["log"]) == len(first["log"]) + 1  # type: ignore[arg-type]


def test_bot_driver_plays_an_all_bot_game_to_the_end() -> None:
    with TestClient(
        create_app(GameRegistry(), bot_delay=0.0, providers=bot_registry())
    ) as client:
        game, _ = _create(client, n_players=2, seats=["random", "random"])
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if client.get(f"/api/games/{game}").json()["status"]["terminal"]:
                break
            time.sleep(0.1)
        body = client.get(f"/api/games/{game}").json()
        assert body["status"]["terminal"] and body["status"]["winner"] is not None


def test_turn_timeout_auto_advances_an_idle_human_turn() -> None:
    # All human, but a turn timeout is set: nobody acts, so the driver auto-
    # plays the idle turn and the game advances on its own.
    with TestClient(create_app(turn_timeout=0.2)) as c:
        game = c.post("/api/games", json={"seed": 0, "seats": ["human"] * 4}).json()[
            "id"
        ]
        deadline = time.monotonic() + 30
        body = c.get(f"/api/games/{game}").json()
        while time.monotonic() < deadline:
            body = c.get(f"/api/games/{game}").json()
            if any(e["kind"] == "move" for e in body["log"]):
                break
            time.sleep(0.05)
        assert any(e["kind"] == "move" for e in body["log"])


def test_no_turn_timeout_leaves_an_idle_human_turn_alone() -> None:
    # Default (no timeout): an all-human game has no driver and never self-plays.
    with TestClient(create_app()) as c:
        game = c.post("/api/games", json={"seed": 0, "seats": ["human"] * 4}).json()[
            "id"
        ]
        time.sleep(0.5)
        body = c.get(f"/api/games/{game}").json()
        assert not any(e["kind"] == "move" for e in body["log"])


def test_concurrent_duplicate_actions_apply_once(client: TestClient) -> None:
    game, tokens = _create(client)
    flat = client.get(f"/api/games/{game}", headers=_hdr(tokens)).json()["actions"][0][
        "flat"
    ]
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
        f"/api/games/{game}/chat",
        json={"text": "hi", "player": 0},
        headers=_hdr(tokens),
    ).json()
    assert body["log"][-1]["kind"] == "chat" and body["log"][-1]["player"] == 0
    # Unowned seat: refused. Spectator (no seat given): allowed.
    assert (
        client.post(
            f"/api/games/{game}/chat", json={"text": "hi", "player": 1}
        ).status_code
        == 403
    )
    assert (
        client.post(f"/api/games/{game}/chat", json={"text": "gl"}).json()["log"][-1][
            "player"
        ]
        is None
    )
    assert (
        client.post(f"/api/games/{game}/chat", json={"text": "   "}).status_code == 422
    )


@contextmanager
def _finished_bot_game(seed: int = 0) -> Iterator[tuple[TestClient, str]]:
    """A fast all-bot app whose game the in-process driver has played to the end,
    yielding the client and the finished game's id."""
    with TestClient(
        create_app(GameRegistry(), bot_delay=0.0, providers=bot_registry())
    ) as client:
        game, _ = _create(client, seed=seed, n_players=2, seats=["random", "random"])
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if client.get(f"/api/games/{game}").json()["status"]["terminal"]:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("game did not finish in time")
        yield client, game


def test_record_and_replay_export_finished_games_only() -> None:
    with TestClient(
        create_app(GameRegistry(), bot_delay=0.0, providers=bot_registry())
    ) as client:
        game, _ = _create(client, n_players=2, seats=["random", "random"])
        # A live game's record would reconstruct hidden hands when replayed.
        assert client.get(f"/api/games/{game}/record").status_code == 409
        assert client.post(f"/api/games/{game}/replay").status_code == 409
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if client.get(f"/api/games/{game}").json()["status"]["terminal"]:
                break
            time.sleep(0.05)
        doc = client.get(f"/api/games/{game}/record").json()
        assert doc["winner"] is not None and len(doc["moves"]) > 0
        opening = client.post(f"/api/games/{game}/replay").json()
        assert opening["move"] == 0 and opening["n_moves"] == len(doc["moves"])
        mid = client.get("/api/replay/state", params={"move": 5}).json()
        assert mid["move"] == 5
        bad = client.get("/api/replay/state", params={"move": 99999})
        assert bad.status_code == 422


def test_replay_upload_roundtrip() -> None:
    with _finished_bot_game() as (client, game):
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
                # An action illegal at the opening (a roll, not a setup move).
                "moves": [{"player": 0, "flat": 126}],
                "winner": None,
            },
        ).status_code
        == 422
    )


def test_oversized_request_body_is_rejected_before_parsing() -> None:
    with TestClient(create_app(GameRegistry(), max_body_bytes=100)) as c:
        big = c.post("/api/replay", json={"pad": "x" * 500})
        assert big.status_code == 413
        # A small body still reaches the route (and is rejected on its merits).
        assert c.post("/api/replay", json={"seed": 1}).status_code == 422


def test_replay_with_too_many_moves_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _finished_bot_game() as (client, game):
        doc = client.get(f"/api/games/{game}/record").json()
        monkeypatch.setattr(
            "settlrl_app.api.routers.replay._MAX_REPLAY_MOVES", len(doc["moves"]) - 1
        )
        resp = client.post("/api/replay", json=doc)
        assert resp.status_code == 422 and "too many moves" in resp.json()["detail"]


def test_get_bots_lists_policies(client: TestClient) -> None:
    body = client.get("/api/bots").json()
    assert "random" in body and body["random"]["description"]
    assert all(
        "counts" in spec and "title" in spec and "description" in spec
        for spec in body.values()
    )


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
