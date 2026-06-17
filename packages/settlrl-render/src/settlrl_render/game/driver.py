"""Server-side game pacing: one asyncio task per game.

It plays due bot moves (sleeping between them so each lands as its own pushed
snapshot for clients to animate), and — when a turn timeout is set — auto-
advances a human turn that has gone idle, so an abandoned game finishes instead
of stalling. The task awaits the game's change event while there is nothing to
do, and exits when the game ends or is evicted. The blocking engine call runs
in a worker thread (``anyio.to_thread``) under the game lock, so it never races
a human request nor stalls the event loop.
"""

import asyncio
import contextlib
import time
from enum import Enum

import anyio.to_thread

from settlrl_render.api.actions import decode_actions
from settlrl_render.api.models import BotMoveModel
from settlrl_render.bots.providers import ProviderRegistry, RemoteBotError
from settlrl_render.game.games import GameHandle
from settlrl_render.game.session import HUMAN, IllegalActionError


class _Due(Enum):
    BOT = "bot"  # a bot seat is acting — play it after the pacing delay
    TIMEOUT = "timeout"  # a human turn has gone idle — auto-play it


class _IdleClock:
    """A per-turn inactivity timer. Any state change (a new version) re-arms it,
    so a player who is actively moving is never timed out; off when timeout<=0."""

    def __init__(self, timeout: float) -> None:
        self.on = timeout > 0
        self._timeout = timeout
        self._deadline: float | None = None
        self._armed_at = -1

    def remaining(self, version: int) -> float:
        """Seconds until the current turn times out, re-arming on a new version;
        <= 0 means it has expired."""
        now = time.monotonic()
        if self._deadline is None or version != self._armed_at:
            self._deadline = now + self._timeout
            self._armed_at = version
        return self._deadline - now

    def reset(self) -> None:
        self._deadline = None


def start_game_driver(
    handle: GameHandle,
    delay: float,
    turn_timeout: float = 0.0,
    providers: ProviderRegistry | None = None,
) -> "asyncio.Task[None]":
    """Schedule the driver for ``handle`` on the running loop (caller tracks the
    task to cancel it on shutdown)."""
    return asyncio.create_task(_drive(handle, delay, turn_timeout, providers))


def _bot_due(handle: GameHandle) -> bool:
    session = handle.session
    return not session.terminal() and session.seats[session.acting_seat()] != HUMAN


def _human_acting(handle: GameHandle) -> bool:
    session = handle.session
    return not session.terminal() and session.seats[session.acting_seat()] == HUMAN


async def _drive(
    handle: GameHandle,
    delay: float,
    turn_timeout: float,
    providers: ProviderRegistry | None,
) -> None:
    clock = _IdleClock(turn_timeout)
    while True:
        due = await _wait_for_due(handle, clock)
        if due is None:
            return  # closed or terminal
        if due is _Due.BOT:
            await asyncio.sleep(delay)  # pace so each move animates on its own
        await _play(handle, clock, due, providers)


async def _wait_for_due(handle: GameHandle, clock: _IdleClock) -> _Due | None:
    """Wait until there is a move to make, returning its kind — or None when the
    game has closed or ended."""
    while True:
        changed = handle._changed  # capture before checking, so no wakeup is lost
        async with handle.lock:
            if handle.closed or handle.session.terminal():
                return None
            timeout: float | None
            if not handle.ready():
                # Waiting in the lobby: do nothing until a claim wakes us.
                clock.reset()
                timeout = None
            elif _bot_due(handle):
                return _Due.BOT
            elif clock.on and _human_acting(handle):
                remaining = clock.remaining(handle.version)
                if remaining <= 0:
                    return _Due.TIMEOUT
                timeout = remaining
            else:
                clock.reset()
                timeout = None
        # On timeout (the idle clock expired) just re-check under the lock.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), timeout=timeout)


async def _bot_move(
    handle: GameHandle, seat: int, providers: ProviderRegistry | None
) -> int | None:
    """The acting bot seat's move, via the seat's remote provider (replay-based).
    A remote failure or an illegal answer falls back to a random legal move, so a
    misbehaving or unregistered service never stalls the game. The blocking engine
    steps run in a worker thread and the remote call is awaited, both under the
    game lock; the seat being a bot's, no human request races it meanwhile."""
    session = handle.session
    remote = providers.remote_for(session.seats[seat]) if providers else None
    if remote is None:
        # No provider for this kind (e.g. unregistered mid-game): keep moving.
        return await anyio.to_thread.run_sync(session.auto_step)
    setup, moves = session.setup.to_dict(), session.moves_flat()
    try:
        flat = int(await remote.act(handle.id, setup, moves, seat))
        await anyio.to_thread.run_sync(session.apply, flat)
        return flat
    except (RemoteBotError, IllegalActionError):
        return await anyio.to_thread.run_sync(session.auto_step)


async def _play(
    handle: GameHandle,
    clock: _IdleClock,
    due: _Due,
    providers: ProviderRegistry | None,
) -> None:
    """Make the due move and push it — unless the moment passed (a human acted
    during the pacing sleep, or just beat the timeout)."""
    async with handle.lock:
        if handle.closed:
            return
        if due is _Due.BOT:
            if not _bot_due(handle):
                return
            seat = handle.session.acting_seat()
            flat = await _bot_move(handle, seat, providers)
        else:
            if not (_human_acting(handle) and clock.remaining(handle.version) <= 0):
                return
            seat = handle.session.acting_seat()
            flat = await anyio.to_thread.run_sync(handle.session.auto_step)
        if flat is not None:
            handle.bot_move = BotMoveModel(
                player=seat, action=decode_actions([flat])[0]
            )
            handle.bump()
    # Re-arm the clock for the next turn (and avoid re-firing if the move was a
    # no-op that left the version unchanged).
    if due is _Due.TIMEOUT:
        clock.reset()
