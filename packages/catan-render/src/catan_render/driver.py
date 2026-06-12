"""Server-side game pacing: a daemon thread per game.

It plays due bot moves (sleeping between them so each lands as its own pushed
snapshot for clients to animate), and — when a turn timeout is set — auto-
advances a human turn that has gone idle, so an abandoned game finishes instead
of stalling. The thread waits on the game's condition while there is nothing to
do, and exits when the game ends or is evicted. Moves are computed under the
lock, so they never race a human request.
"""

import threading
import time
from collections.abc import Callable
from enum import Enum

from .actions import decode_actions
from .games import GameHandle
from .models import BotMoveModel
from .session import HUMAN


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
    handle: GameHandle, delay: float, turn_timeout: float = 0.0
) -> None:
    threading.Thread(
        target=_drive, args=(handle, delay, turn_timeout), daemon=True
    ).start()


def _bot_due(handle: GameHandle) -> bool:
    session = handle.session
    return not session.terminal() and session.seats[session.acting_seat()] != HUMAN


def _human_acting(handle: GameHandle) -> bool:
    session = handle.session
    return not session.terminal() and session.seats[session.acting_seat()] == HUMAN


def _drive(handle: GameHandle, delay: float, turn_timeout: float) -> None:
    clock = _IdleClock(turn_timeout)
    while True:
        due = _wait_for_due(handle, clock)
        if due is None:
            return  # closed or terminal
        if due is _Due.BOT:
            time.sleep(delay)  # pace so each move animates as its own snapshot
        _play(handle, clock, due)


def _wait_for_due(handle: GameHandle, clock: _IdleClock) -> _Due | None:
    """Block until there is a move to make, returning its kind — or None when
    the game has closed or ended."""
    with handle.lock:
        while True:
            if handle.closed or handle.session.terminal():
                return None
            if _bot_due(handle):
                return _Due.BOT
            if clock.on and _human_acting(handle):
                remaining = clock.remaining(handle.version)
                if remaining <= 0:
                    return _Due.TIMEOUT
                handle.lock.wait(timeout=remaining)
            else:
                clock.reset()
                handle.lock.wait()


def _play(handle: GameHandle, clock: _IdleClock, due: _Due) -> None:
    """Make the due move and push it — unless the moment passed (a human acted
    during the pacing sleep, or just beat the timeout)."""
    with handle.lock:
        if handle.closed:
            return
        step: Callable[[], int | None]
        if due is _Due.BOT:
            if not _bot_due(handle):
                return
            step = handle.session.bot_step
        else:
            if not (_human_acting(handle) and clock.remaining(handle.version) <= 0):
                return
            step = handle.session.auto_step
        seat = handle.session.acting_seat()
        flat = step()
        if flat is not None:
            handle.bot_move = BotMoveModel(
                player=seat, action=decode_actions([flat])[0]
            )
            handle.bump()
    # Re-arm the clock for the next turn (and avoid re-firing if the move was a
    # no-op that left the version unchanged).
    if due is _Due.TIMEOUT:
        clock.reset()
