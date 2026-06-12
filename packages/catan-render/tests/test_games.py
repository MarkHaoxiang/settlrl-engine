"""Unit tests for the game registry and seat claims (no engine, no HTTP)."""

import time
from typing import cast

import pytest
from catan_render.games import (
    GameHandle,
    GameRegistry,
    QueuePosition,
    RegistryFullError,
)
from catan_render.session import GameSession

# A touch time well past every eviction TTL. Relative to ``monotonic()`` rather
# than 0.0, which is only "ancient" when the clock's origin is far in the past
# (it isn't on a freshly booted CI runner).
_LONG_AGO = time.monotonic() - 100_000


class _FakeSession:
    """Just enough session for registry logic: seat kinds, terminality, and
    whether any move has been played (default 1 = a started game)."""

    def __init__(
        self, seats: list[str], terminal: bool = False, moves_played: int = 1
    ) -> None:
        self.seats = seats
        self._terminal = terminal
        self.moves_played = moves_played

    def terminal(self) -> bool:
        return self._terminal


def _session(
    seats: list[str] | None = None, terminal: bool = False, moves_played: int = 1
) -> GameSession:
    return cast(
        GameSession,
        _FakeSession(
            seats or ["human", "human", "random", "random"], terminal, moves_played
        ),
    )


def test_claim_defaults_to_first_free_human_seat() -> None:
    handle = GameHandle("g", _session())
    seat, token = handle.claim()
    assert seat == 0 and token
    seat2, token2 = handle.claim()
    assert seat2 == 1 and token2 != token
    with pytest.raises(LookupError):
        handle.claim()


def test_claim_rejects_bots_and_taken_seats() -> None:
    handle = GameHandle("g", _session())
    handle.claim(0)
    with pytest.raises(ValueError, match="already claimed"):
        handle.claim(0)
    with pytest.raises(ValueError, match="not a human seat"):
        handle.claim(2)


def test_owned_seats_requires_exact_tokens() -> None:
    handle = GameHandle("g", _session())
    _, t0 = handle.claim(0)
    _, t1 = handle.claim(1)
    assert handle.owned_seats([t0]) == {0}
    assert handle.owned_seats([t1, t0]) == {0, 1}
    assert handle.owned_seats(["bogus"]) == set()
    assert handle.owned_seats([]) == set()


def test_registry_creates_unique_ids_and_resolves_them() -> None:
    registry = GameRegistry()
    a = registry.create(_session())
    b = registry.create(_session())
    assert a.id != b.id
    assert registry.get(a.id) is a
    assert registry.get("nope") is None


def test_eviction_prefers_finished_then_abandoned() -> None:
    registry = GameRegistry(max_games=2)
    running = registry.create(_session())
    finished = registry.create(_session(terminal=True))
    registry.create(_session())  # evicts the finished game first
    assert registry.get(finished.id) is None
    assert registry.get(running.id) is running

    # A running game idle past the TTL counts as abandoned and goes next.
    running.touched = _LONG_AGO
    registry.create(_session())
    assert registry.get(running.id) is None


def test_full_registry_of_active_games_refuses_creation() -> None:
    registry = GameRegistry(max_games=1)
    active = registry.create(_session())
    with pytest.raises(RegistryFullError):
        registry.create(_session())
    assert registry.get(active.id) is active


def test_unstarted_idle_game_is_reclaimed_before_a_played_one() -> None:
    # A create-flood leftover: a game no one moved in, idle past the short
    # grace. It must yield its slot even though a started game is younger.
    registry = GameRegistry(max_games=2)
    started = registry.create(_session(moves_played=3))
    unstarted = registry.create(_session(moves_played=0))
    unstarted.touched = _LONG_AGO  # idle well past the unstarted grace
    registry.create(_session())
    assert registry.get(unstarted.id) is None
    assert registry.get(started.id) is started


def test_unstarted_but_recent_game_is_protected() -> None:
    # An unstarted game someone just created (about to join) is not a leftover.
    registry = GameRegistry(max_games=1)
    fresh = registry.create(_session(moves_played=0))
    with pytest.raises(RegistryFullError):
        registry.create(_session())
    assert registry.get(fresh.id) is fresh


def _finish(handle: GameHandle) -> None:
    cast("_FakeSession", handle.session)._terminal = True


def test_admit_seats_until_the_active_cap_then_queues_fifo() -> None:
    reg = GameRegistry(max_games=10, max_active=1)
    seated = reg.admit(_session(), None)
    assert isinstance(seated, GameHandle)  # first one is seated immediately

    # total is the queue length at the moment each creator was told their place.
    second = reg.admit(_session(), None)
    assert isinstance(second, QueuePosition) and (second.position, second.total) == (
        1,
        1,
    )
    third = reg.admit(_session(), None)
    assert isinstance(third, QueuePosition) and (third.position, third.total) == (2, 2)

    # Re-polling holds your place; the head is not seated while the cap is full.
    again = reg.admit(_session(), second.ticket)
    assert isinstance(again, QueuePosition) and again.position == 1

    # A freed slot seats the head, and only the head.
    _finish(seated)
    head = reg.admit(_session(), second.ticket)
    assert isinstance(head, GameHandle)
    behind = reg.admit(_session(), third.ticket)
    assert isinstance(behind, QueuePosition) and behind.position == 1  # now the head


def test_fresh_request_never_jumps_a_waiting_queue() -> None:
    reg = GameRegistry(max_games=10, max_active=1)
    seated = reg.admit(_session(), None)
    assert isinstance(seated, GameHandle)
    waiting = reg.admit(_session(), None)
    assert isinstance(waiting, QueuePosition)
    _finish(seated)  # a slot is free, but someone is already in line
    fresh = reg.admit(_session(), None)
    assert isinstance(fresh, QueuePosition) and fresh.position == 2  # behind the waiter


def test_abandoned_ticket_ages_out_of_the_queue() -> None:
    reg = GameRegistry(max_games=10, max_active=1)
    reg.admit(_session(), None)  # seats the one active slot
    ghost = reg.admit(_session(), None)
    assert isinstance(ghost, QueuePosition)
    reg._queue[0].last_seen = _LONG_AGO  # the ghost stopped polling
    nxt = reg.admit(_session(), None)  # a new creator: the ghost is pruned first
    assert isinstance(nxt, QueuePosition) and nxt.total == 1
