"""Server-side game pacing: a daemon thread per game.

The thread plays due bot moves (sleeping between them so each lands as its own
pushed snapshot for clients to animate), and — when a turn timeout is set —
auto-advances a human turn that has gone idle, so an abandoned game finishes
instead of stalling forever. It waits on the game's condition while there is
nothing to do, and exits when the game ends or is evicted. Pacing sleeps and
the timeout wait happen outside nothing-to-do windows under the lock; the bot
move itself is computed under the lock, so it never races a human request.
"""

import threading
import time

from .actions import decode_actions
from .games import GameHandle
from .models import BotMoveModel
from .session import HUMAN


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


def _record_move(handle: GameHandle, seat: int, flat: int) -> None:
    handle.bot_move = BotMoveModel(player=seat, action=decode_actions([flat])[0])
    handle.bump()


def _drive(handle: GameHandle, delay: float, turn_timeout: float) -> None:
    # The human-turn deadline is an inactivity clock: any state change (a new
    # version) re-arms it, so a player actively moving is never timed out.
    deadline: float | None = None
    deadline_version = -1
    while True:
        due = ""
        with handle.lock:
            while True:
                if handle.closed or handle.session.terminal():
                    return
                if _bot_due(handle):
                    due = "bot"
                    break
                if turn_timeout > 0 and _human_acting(handle):
                    now = time.monotonic()
                    if deadline is None or handle.version != deadline_version:
                        deadline = now + turn_timeout
                        deadline_version = handle.version
                    if now >= deadline:
                        due = "timeout"
                        break
                    handle.lock.wait(timeout=deadline - now)
                else:
                    deadline = None
                    handle.lock.wait()
        if due == "bot":
            time.sleep(delay)
            with handle.lock:
                if handle.closed or not _bot_due(handle):
                    continue
                seat = handle.session.acting_seat()
                flat = handle.session.bot_step()
                if flat is not None:
                    _record_move(handle, seat, flat)
        else:  # timeout
            with handle.lock:
                # The human may have acted between waking and re-acquiring.
                if handle.closed or not _human_acting(handle):
                    deadline = None
                    continue
                if deadline is None or time.monotonic() < deadline:
                    continue
                seat = handle.session.acting_seat()
                flat = handle.session.auto_step()
                if flat is not None:
                    _record_move(handle, seat, flat)
            deadline = None
