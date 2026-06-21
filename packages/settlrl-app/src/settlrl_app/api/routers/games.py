"""Game routes: create / join / read / act / event-stream / chat / record, plus
loading a finished game into the replay tooling.

Routes stay thin: resolve the game, check seat tokens, hold the per-game lock,
offload the blocking engine step, map errors to status codes. Game logic lives
in ``session``, seat claims in ``games``, and what a requester may see in
``views``.
"""

from collections.abc import AsyncIterator
from random import Random
from typing import Annotated, Literal

import anyio.to_thread
import settlrl_game.reference as ref
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from settlrl_game.convert import board_to_model
from settlrl_game.models import BoardModel, GameModel, ReplayStateModel
from settlrl_game.record import GameRecord
from settlrl_game.session import HUMAN, GameSession, IllegalActionError
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response

from settlrl_app.api.deps import Deps, SeatTokens, needs_driver, tokens, uid
from settlrl_app.api.routers.replay import load_replay
from settlrl_app.api.views import game_model
from settlrl_app.game.games import (
    GameHandle,
    QueuePosition,
    RegistryFullError,
    win_threshold,
)
from settlrl_app.storage.db import User
from settlrl_app.storage.store import GameStore


class _CreateRequest(BaseModel):
    seed: int = 0
    # Seats in the new game. The engine supports 2-4; the app offers 2
    # and 4 for now.
    n_players: Literal[2, 4] = 4
    number_placement: Literal["random", "spiral"] = "random"
    # What controls each seat: "human" or a bot kind from GET /api/bots; no seat
    # has to be human. None seats a human on seat 0 and "random" bots elsewhere.
    seats: list[str] | None = None
    # Human seats the creator claims immediately: every one ("all", the
    # hotseat default), just the first ("first", online play — others join
    # via POST /join), or none.
    claim: Literal["all", "first", "none"] = "all"
    # List the game in the public lobby (anyone may join an open human seat).
    # Default off: a game is invite-only unless explicitly made public.
    listed: bool = False
    # Mark the game open to Quick Match (a visibility flag shown in the lobby).
    searchable: bool = False
    # VP total that ends the game; None uses the server default for the count
    # (win_threshold: 15 for the 2-player duel, 10 otherwise).
    victory_points_to_win: int | None = None
    # The caller's place in line from a prior queued response, re-sent each poll
    # while waiting for a free slot; None on the first attempt.
    ticket: str | None = None


class _CreatedModel(BaseModel):
    """A fresh game: its id and the creator's seat tokens."""

    id: str
    seats: list[str]
    tokens: dict[int, str]


class _QueuedModel(BaseModel):
    """The server is at its concurrency cap: the caller's place in line. They
    re-POST with ``ticket`` until they get a :class:`_CreatedModel` back."""

    queued: Literal[True] = True
    ticket: str
    position: int
    total: int


class _JoinRequest(BaseModel):
    # A specific human seat, or None for the first unclaimed one.
    seat: int | None = None


class _JoinedModel(BaseModel):
    id: str
    seat: int
    token: str


class _SeatRequest(BaseModel):
    # The seat to retarget (must be unclaimed) and its new controller: "human"
    # (open for a joiner) or a bot kind from GET /api/bots.
    seat: int
    kind: str


class _ConfigureRequest(BaseModel):
    """A pre-start reconfiguration of the lobby room's game. Every field is
    optional and merged onto the current setup; the board is rebuilt in place
    (the game id, the still-valid seat claims, and the chat are preserved)."""

    seed: int | None = None
    n_players: Literal[2, 4] | None = None
    number_placement: Literal["random", "spiral"] | None = None
    # The full seat-kind list (length == the new n_players); a still-claimed
    # surviving seat is forced back to "human" regardless.
    seats: list[str] | None = None
    victory_points_to_win: int | None = None
    listed: bool | None = None
    searchable: bool | None = None


class _LeftModel(BaseModel):
    """The outcome of leaving: the whole lobby was closed (the host left) or just
    the caller's seat was freed."""

    closed: bool


class _ActionRequest(BaseModel):
    flat: int


class _ChatRequest(BaseModel):
    text: str
    # Seat the message belongs to (must be owned); None for a spectator.
    player: int | None = None


def _validate_seat_kinds(
    seats: "list[str] | None",
    n_players: int,
    catalog: dict[str, dict[str, object]],
) -> None:
    """Reject a create whose bot seats name unknown kinds or kinds that don't
    support the player count, against the registered bot catalog (the game server
    runs no bots, so a seat kind is valid only if some provider serves it)."""
    if seats is None:
        return
    for kind in seats:
        if kind == HUMAN:
            continue
        spec = catalog.get(kind)
        if spec is None:
            raise HTTPException(status_code=422, detail=f"unknown bot kind: {kind!r}")
        counts = spec.get("counts", [])
        if isinstance(counts, list) and n_players not in counts:
            raise HTTPException(
                status_code=422,
                detail=f"{kind} is not available in a {n_players}-player game",
            )


def _rejournal(store: GameStore, handle: GameHandle) -> None:
    """Rewrite an unstarted game's journal from its current claims and chat. The
    journal records claims additively, so freeing a seat is expressed by
    rebuilding (the header write is an upsert). Safe pre-start only: no moves are
    journalled yet, so none are lost."""
    handle.journal = store.create(handle.id, handle.session.setup.to_dict())
    for seat, token in handle.claims.items():
        handle.journal.claim(seat, token, handle.claim_users.get(seat))
    for entry in handle.session.log():
        if entry.kind == "chat":
            handle.journal.chat(entry.player, entry.text)


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    registry, bots = deps.registry, deps.bots
    CurrentUser = Annotated[User | None, Depends(deps.auth.optional_user)]

    @router.get("/api/preview")
    def get_preview(
        seed: int = 0,
        n_players: int = 4,
        number_placement: Literal["random", "spiral"] = "random",
    ) -> BoardModel:
        """The board a new game would open on, for the map picker — no game is
        created. Terrain and ports depend only on the seed (not the placement)."""
        if n_players not in (2, 3, 4):
            raise HTTPException(status_code=422, detail="n_players must be 2-4")
        layout = ref.random_layout(Random(seed), number_placement)
        game = ref.Game.new(layout, ref.desert_tile(layout), n_players=n_players)
        return board_to_model(game)

    @router.post("/api/games")
    async def post_create(
        req: _CreateRequest, response: Response, user: CurrentUser = None
    ) -> _CreatedModel | _QueuedModel:
        """Create a game, or return the caller's place in line when the server
        is at its concurrency cap (a ``202`` they re-POST with ``ticket``).
        Listing a game publicly (``listed``) requires a signed-in account."""
        if (req.listed or req.searchable) and user is None:
            raise HTTPException(
                status_code=401, detail="sign in to list a game in the lobby"
            )
        deps.guard_one_game(user)
        catalog = bots.catalog()
        _validate_seat_kinds(req.seats, req.n_players, catalog)
        try:
            session = GameSession(seed=req.seed, n_players=req.n_players)
            await anyio.to_thread.run_sync(
                lambda: session.reset(
                    req.seed,
                    number_placement=req.number_placement,
                    seats=req.seats,
                    external_kinds=bots.remote_kinds(),
                    victory_points_to_win=(
                        req.victory_points_to_win or win_threshold(req.n_players)
                    ),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            seated = registry.admit(session, req.ticket)
        except RegistryFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(seated, QueuePosition):
            response.status_code = 202
            return _QueuedModel(
                ticket=seated.ticket, position=seated.position, total=seated.total
            )
        seated.listed = req.listed  # show in the public lobby if requested
        seated.searchable = req.searchable  # mark open to Quick Match (visibility)
        humans = seated.human_seats()
        claiming = (
            humans if req.claim == "all" else humans[:1] if req.claim == "first" else []
        )
        game_tokens = dict(seated.claim(seat, uid(user)) for seat in claiming)
        if user is not None:
            label = user.email.split("@", 1)[0]
            for seat in claiming:
                seated.claim_names[seat] = label
        if needs_driver(seated, deps.turn_timeout):
            deps.spawn_driver(seated)
        return _CreatedModel(id=seated.id, seats=session.seats, tokens=game_tokens)

    @router.post("/api/games/{game_id}/join")
    async def post_join(
        game_id: str, req: _JoinRequest, user: CurrentUser = None
    ) -> _JoinedModel:
        handle = deps.handle_of(game_id)
        deps.guard_one_game(user, allow=game_id)
        async with handle.lock:
            try:
                seat, token = handle.claim(req.seat, uid(user))
            except (LookupError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if user is not None:
                handle.claim_names[seat] = user.email.split("@", 1)[0]
            handle.bump()
        return _JoinedModel(id=game_id, seat=seat, token=token)

    @router.post("/api/games/{game_id}/leave")
    async def post_leave(
        game_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> _LeftModel:
        """Leave a lobby that hasn't started. The host (seat-0 owner) closes the
        whole game and everyone else is bounced; any other participant just frees
        their own seat back to open. ``403`` if you hold no seat here, ``409``
        once the game has started (an in-progress game isn't closable)."""
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if not owned:
                raise HTTPException(status_code=403, detail="you are not in this game")
            if handle.ready():
                raise HTTPException(
                    status_code=409, detail="the game has already started"
                )
            if 0 in owned:
                registry.remove(game_id)
                return _LeftModel(closed=True)
            for seat in owned:
                handle.release_seat(seat)
            if deps.store is not None and handle.journal is not None:
                _rejournal(deps.store, handle)
            handle.bump()
            return _LeftModel(closed=False)

    @router.post("/api/games/{game_id}/seats")
    async def post_seat(
        game_id: str,
        req: _SeatRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> GameModel:
        """Retarget an unclaimed seat before play begins — open it for a human or
        turn it into a bot (the lobby owner's "fill to start" control). Any
        player in the game may do this; ``403`` for outsiders, ``409`` if the
        seat is taken or the game has started."""
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if not owned:
                raise HTTPException(
                    status_code=403, detail="only a player in this game can set seats"
                )
            if req.seat in handle.claims:
                raise HTTPException(status_code=409, detail="seat is already taken")
            if req.kind != HUMAN:
                _validate_seat_kinds(
                    [req.kind], handle.session.n_players, bots.catalog()
                )
            try:
                handle.session.set_seat_kind(req.seat, req.kind)
            except (ValueError, IllegalActionError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if needs_driver(handle, deps.turn_timeout):
                deps.spawn_driver(handle)
            handle.bump()
            return game_model(handle, owned)

    @router.post("/api/games/{game_id}/configure")
    async def post_configure(
        game_id: str,
        req: _ConfigureRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> GameModel:
        """Reconfigure a game that hasn't started — change the map, player count,
        VP target, seats, or lobby flags. The board is rebuilt in place, keeping
        the game id, the still-valid seat claims, and the chat. Host only (the
        seat-0 owner): ``403`` otherwise; ``409`` once a move has been played."""
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if 0 not in owned:
                raise HTTPException(
                    status_code=403, detail="only the host can configure"
                )
            session = handle.session
            if session.moves_played > 0:
                raise HTTPException(status_code=409, detail="game already started")

            n = req.n_players if req.n_players is not None else session.n_players
            if req.seats is not None:
                seats = list(req.seats)
            else:
                # Growing the count opens the new seats for humans — never assume
                # a bot kind exists (no "random" service may be registered); the
                # host fills them from the live catalog afterwards.
                cur = session.seats
                seats = [cur[i] if i < len(cur) else HUMAN for i in range(n)]
            if len(seats) != n:
                raise HTTPException(status_code=422, detail=f"expected {n} seats")
            # A claimed seat that survives the new count stays human — you can't
            # turn a seat someone holds into a bot.
            for seat in handle.claims:
                if seat < n:
                    seats[seat] = HUMAN
            _validate_seat_kinds(seats, n, bots.catalog())

            # Snapshot what survives the reset (which clears the session's log).
            chat = [(e.player, e.text) for e in session.log() if e.kind == "chat"]
            old_claims = dict(handle.claims)
            old_users = dict(handle.claim_users)
            old_names = dict(handle.claim_names)
            vp = req.victory_points_to_win or session.victory_points_to_win
            placement = req.number_placement or session.number_placement
            seed = req.seed if req.seed is not None else session.seed
            try:
                await anyio.to_thread.run_sync(
                    lambda: session.reset(
                        seed,
                        n_players=n,
                        number_placement=placement,
                        seats=seats,
                        external_kinds=bots.remote_kinds(),
                        victory_points_to_win=vp,
                    )
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            # Keep claims only for seats that still exist and are still human.
            handle.claims, handle.claim_users, handle.claim_names = {}, {}, {}
            for seat, token in old_claims.items():
                if seat < n and seats[seat] == HUMAN:
                    handle.claims[seat] = token
                    if seat in old_users:
                        handle.claim_users[seat] = old_users[seat]
                    if seat in old_names:
                        handle.claim_names[seat] = old_names[seat]
            kept_chat = [(p, t) for p, t in chat if p is None or p < n]
            for player, text in kept_chat:
                session.add_chat(player, text)

            if req.listed is not None:
                handle.listed = req.listed
            if req.searchable is not None:
                handle.searchable = req.searchable

            # Rebuild the journal so a crash-restart restores the new board,
            # surviving claims, and chat (the header write is an upsert).
            if deps.store is not None:
                handle.journal = deps.store.create(game_id, session.setup.to_dict())
                for seat, token in handle.claims.items():
                    handle.journal.claim(seat, token, handle.claim_users.get(seat))
                for player, text in kept_chat:
                    handle.journal.chat(player, text)

            if needs_driver(handle, deps.turn_timeout):
                deps.spawn_driver(handle)
            handle.bump()
            # Recompute ownership: a dropped claim (a now-gone seat) must not
            # linger in the snapshot's your_seats.
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            return game_model(handle, owned)

    @router.get("/api/games/{game_id}")
    async def get_game(
        game_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> GameModel:
        handle = deps.handle_of(game_id)
        async with handle.lock:
            return game_model(
                handle, handle.owned_seats(tokens(x_seat_tokens), uid(user))
            )

    @router.post("/api/games/{game_id}/action")
    async def post_action(
        game_id: str,
        req: _ActionRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> GameModel:
        """Apply the acting seat's move; the request must prove that seat."""
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if handle.session.acting_seat() not in owned:
                raise HTTPException(status_code=403, detail="not your seat")
            try:
                await anyio.to_thread.run_sync(handle.session.apply, req.flat)
            except IllegalActionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            handle.bot_move = None
            handle.bump()
            return game_model(handle, owned)

    @router.get("/api/games/{game_id}/events")
    async def get_events(
        game_id: str, user: CurrentUser = None, x_seat_tokens: SeatTokens = None
    ) -> EventSourceResponse:
        """Server-sent events: the requester's snapshot now, then again on every
        state change (moves, bot plays, chat, joins). ``EventSourceResponse``
        adds the SSE framing, keepalive pings, and client-disconnect teardown."""
        handle = deps.handle_of(game_id)
        seat_tokens = tokens(x_seat_tokens)
        user_id = uid(user)

        async def stream() -> AsyncIterator[str]:
            seen = -1
            while True:
                changed = handle._changed  # capture before serialising/waiting
                async with handle.lock:
                    if handle.closed:
                        return
                    if handle.version != seen:
                        seen = handle.version
                        body: str | None = game_model(
                            handle, handle.owned_seats(seat_tokens, user_id)
                        ).model_dump_json()
                    else:
                        body = None
                if body is not None:
                    yield body  # EventSourceResponse wraps it as a `data:` event
                else:
                    await changed.wait()  # idle: pings come from EventSourceResponse

        return EventSourceResponse(stream(), ping=15)

    @router.post("/api/games/{game_id}/chat")
    async def post_chat(
        game_id: str,
        req: _ChatRequest,
        user: CurrentUser = None,
        x_seat_tokens: SeatTokens = None,
    ) -> GameModel:
        """Append a chat message to the game log."""
        text = req.text.strip()
        if not text or len(text) > 500:
            raise HTTPException(
                status_code=422, detail="chat text must be 1-500 characters"
            )
        handle = deps.handle_of(game_id)
        async with handle.lock:
            owned = handle.owned_seats(tokens(x_seat_tokens), uid(user))
            if req.player is not None and req.player not in owned:
                raise HTTPException(status_code=403, detail="not your seat")
            handle.session.add_chat(req.player, text)
            if handle.journal is not None:
                handle.journal.chat(req.player, text)
            handle.bump()
            return game_model(handle, owned)

    async def _finished_record(game_id: str) -> GameRecord:
        """A finished game's replayable record — from the live handle, or rebuilt
        from the journal store when the handle has been evicted. 409 while still
        running (replaying reconstructs hidden hands); 404 if unknown."""
        handle = deps.registry.get(game_id)
        if handle is not None:
            async with handle.lock:
                if not handle.session.terminal():
                    raise HTTPException(status_code=409, detail="game still running")
                return handle.session.record()
        if deps.store is not None:
            record = await deps.store.finished_record(game_id)
            if record is not None:
                return record
        raise HTTPException(status_code=404, detail="no such game")

    @router.get("/api/games/{game_id}/record")
    async def get_record(game_id: str) -> Response:
        """The finished game as ``GameRecord`` JSON -- a self-contained,
        replayable transcript (served for past games too, rebuilt from the
        store). 409 while running; 404 if unknown."""
        record = await _finished_record(game_id)
        return Response(content=record.to_json(), media_type="application/json")

    @router.post("/api/games/{game_id}/replay")
    async def post_replay_from_game(game_id: str) -> ReplayStateModel:
        """Load a finished game for replay (a past game too, rebuilt from the
        store). 409 while it is still running; 404 if unknown."""
        return await load_replay(deps.replays, await _finished_record(game_id))

    return router
