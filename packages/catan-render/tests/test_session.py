"""Tests for :class:`session.GameSession` turn flow.

The renderer plays seat 0 (the human); the other seats are auto-played by random
bots after each human move. These check that the acting seat after construction /
reset is the human, that bots advance the game back to the human after a move,
and that a full game can be driven to a terminal state with a winner.
"""

from catan_render.session import HUMAN_SEAT, GameSession


def test_acting_seat_is_human_after_construction() -> None:
    sess = GameSession(seed=0)
    # After construction the bots (if any opened) have run; it's the human's turn.
    assert sess.acting_seat() == HUMAN_SEAT
    assert sess.status().your_turn
    assert not sess.status().terminal


def test_acting_seat_is_human_after_reset() -> None:
    sess = GameSession(seed=1)
    sess.apply(int(sess.legal_flat()[0]))  # perturb
    sess.reset(seed=2)
    assert sess.acting_seat() == HUMAN_SEAT
    assert sess.status().your_turn


def test_bots_auto_advance_after_human_move() -> None:
    # After a single human move the session runs bots until it is the human's
    # turn again (or the game ends): the acting seat is the human once more.
    sess = GameSession(seed=0)
    legal = sess.legal_flat()
    assert legal.size > 0
    sess.apply(int(legal[0]))
    assert sess.terminal() or sess.acting_seat() == HUMAN_SEAT


def test_run_bots_is_idempotent_when_human_acting() -> None:
    # When it's already the human's turn, driving the bots changes nothing.
    sess = GameSession(seed=0)
    assert sess.acting_seat() == HUMAN_SEAT
    before = sess.legal_flat().tolist()
    sess._run_bots()
    assert sess.acting_seat() == HUMAN_SEAT
    assert sess.legal_flat().tolist() == before


def test_game_drives_to_completion() -> None:
    # The human also plays a random legal move each turn; the game must reach a
    # terminal state with a real winner within a sane step budget.
    sess = GameSession(seed=0)
    rng = sess._rng
    for _ in range(50_000):
        if sess.terminal():
            break
        legal = sess.legal_flat()
        if legal.size == 0:
            break
        sess.apply(int(rng.choice(legal)))
    status = sess.status()
    assert status.terminal
    assert status.winner is not None
    assert 0 <= status.winner < 4


def test_two_player_session() -> None:
    # n_players=2 seats the human and one bot, and a random game still drives
    # to completion with a seated winner.
    sess = GameSession(seed=0, n_players=2)
    assert sess.acting_seat() == HUMAN_SEAT
    assert len(sess.board[1].player_resources[0]) == 2  # player axis = seats
    rng = sess._rng
    for _ in range(50_000):
        if sess.terminal():
            break
        legal = sess.legal_flat()
        if legal.size == 0:
            break
        assert sess.acting_seat() in (0, 1)
        sess.apply(int(rng.choice(legal)))
    status = sess.status()
    assert status.terminal
    assert status.winner is not None and 0 <= status.winner < 2


def test_reset_keeps_seat_count_unless_changed() -> None:
    sess = GameSession(seed=0, n_players=2)
    sess.reset(seed=1)  # no n_players -> keeps 2
    assert sess.n_players == 2
    sess.reset(seed=2, n_players=4)
    assert sess.n_players == 4
