"""Unit tests for the game registry and seat claims (no engine, no HTTP)."""

from typing import cast

import pytest
from catan_render.games import GameHandle, GameRegistry
from catan_render.session import GameSession


class _FakeSession:
    """Just enough session for registry logic: seat kinds + terminality."""

    def __init__(self, seats: list[str], terminal: bool = False) -> None:
        self.seats = seats
        self._terminal = terminal

    def terminal(self) -> bool:
        return self._terminal


def _session(seats: list[str] | None = None, terminal: bool = False) -> GameSession:
    return cast(GameSession, _FakeSession(seats or ["human", "human", "random", "random"], terminal))


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


def test_eviction_prefers_finished_then_least_recently_touched() -> None:
    registry = GameRegistry(max_games=2)
    running = registry.create(_session())
    finished = registry.create(_session(terminal=True))
    registry.create(_session())  # evicts the finished game first
    assert registry.get(finished.id) is None
    assert registry.get(running.id) is running

    # Both running now: the least recently touched goes.
    other = registry.create(_session())  # at cap again
    running.touched = 0.0
    assert other is not None
    registry.create(_session())
    assert registry.get(running.id) is None
