"""Server-side bot pacing: a daemon thread per game plays due bot moves.

The thread sleeps between moves so each one lands as its own pushed snapshot
(clients animate per move), waits on the game's condition while a human is
thinking, and exits when the game ends or is evicted. Pacing sleeps happen
outside the lock, so human requests are never blocked by it.
"""

import threading
import time

from .actions import decode_actions
from .games import GameHandle
from .models import BotMoveModel
from .session import HUMAN


def start_bot_driver(handle: GameHandle, delay: float) -> None:
    threading.Thread(target=_drive, args=(handle, delay), daemon=True).start()


def _bot_due(handle: GameHandle) -> bool:
    session = handle.session
    return not session.terminal() and session.seats[session.acting_seat()] != HUMAN


def _drive(handle: GameHandle, delay: float) -> None:
    while True:
        with handle.lock:
            while not handle.closed and not _bot_due(handle):
                if handle.session.terminal():
                    return
                handle.lock.wait()
            if handle.closed:
                return
        time.sleep(delay)
        with handle.lock:
            if handle.closed or not _bot_due(handle):
                continue
            seat = handle.session.acting_seat()
            flat = handle.session.bot_step()
            if flat is not None:
                handle.bot_move = BotMoveModel(
                    player=seat, action=decode_actions([flat])[0]
                )
                handle.bump()
