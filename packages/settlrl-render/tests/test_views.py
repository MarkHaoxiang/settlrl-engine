"""Per-seat view tests: redaction and turn scoping, without HTTP.

``game_model`` is the hidden-information boundary, so beyond the spot checks
there is a sweep: drive a random hotseat game and assert that while it is
running no observer's view carries another player's hand, off-turn actions, or
a belief that isn't theirs. Once the game ends every hand is revealed to
everyone (see ``test_terminal_reveals_every_hand``).
"""

import numpy as np
from _helpers import BOT_KINDS
from settlrl_render.api.views import game_model
from settlrl_render.game.games import GameHandle
from settlrl_render.game.session import GameSession


def _handle(session: GameSession) -> GameHandle:
    return GameHandle("g", session)


def test_owner_sees_own_hand_actions_and_belief() -> None:
    handle = _handle(GameSession(seed=0))  # all-human (seat 0 opens)
    view = game_model(handle, owned={0})
    assert view.status.your_turn and view.actions
    players = {p.player: p for p in view.board.players}
    assert players[0].resources is not None
    assert all(players[p].resources is None for p in (1, 2, 3))
    assert view.belief is not None and view.belief.observer == 0


def test_spectator_sees_public_counts_only() -> None:
    handle = _handle(GameSession(seed=0))
    view = game_model(handle, owned=set())
    assert not view.status.your_turn and view.actions == []
    assert view.belief is None
    assert all(p.resources is None for p in view.board.players)
    # Public counts survive redaction.
    assert all(p.resource_cards >= 0 for p in view.board.players)


def test_belief_follows_the_acting_owned_seat() -> None:
    # Hotseat owning two seats: the view observes through whichever owned
    # seat is acting (seat 0 opens the game).
    handle = _handle(
        GameSession(
            seed=0,
            seats=["human", "human", "random", "random"],
            external_kinds=BOT_KINDS,
        )
    )
    view = game_model(handle, owned={0, 1})
    assert view.status.your_turn
    assert view.belief is not None and view.belief.observer == 0
    players = {p.player: p for p in view.board.players}
    assert players[0].resources is not None and players[1].resources is not None
    assert players[2].resources is None


def test_running_views_never_leak_hidden_hands() -> None:
    # A random 4-human game, checked at every step (while running) for every
    # observer kind. Redaction only applies before the game ends.
    session = GameSession(seed=1, seats=["human"] * 4)
    handle = _handle(session)
    rng = np.random.default_rng(0)
    for _ in range(80):
        if session.terminal():
            break
        acting = session.acting_seat()
        for owned in (set(), {0}, {2}, {1, 3}):
            doc = game_model(handle, owned).model_dump()
            for p in doc["board"]["players"]:
                if p["player"] not in owned:
                    assert p["resources"] is None
                    assert p["dev_card_types"] is None
            if acting not in owned:
                assert doc["actions"] == []
                assert not doc["status"]["your_turn"]
            if owned:
                assert doc["belief"]["observer"] in owned
            else:
                assert doc["belief"] is None
        session.apply(int(rng.choice(session.legal_flat())))


def test_terminal_reveals_every_hand() -> None:
    # When the game is over, every hand is open — even to a spectator.
    session = GameSession(seed=3, n_players=2, seats=["human", "human"])
    while not session.terminal():  # play it out to a winner with random moves
        session.auto_step()
    assert session.terminal()
    view = game_model(_handle(session), owned=set())
    assert all(
        p.resources is not None and p.dev_card_types is not None
        for p in view.board.players
    )
