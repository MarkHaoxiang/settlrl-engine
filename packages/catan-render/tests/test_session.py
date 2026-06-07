"""Tests for :class:`session.GameSession` turn flow.

Each seat is configured per game as a human (hotseat) or a bot (a catan-agents
policy); no seat has to be human. These check that ``bot_step`` advances bot
seats one move at a time (and only bot seats), that an all-bot game plays
itself, and that a full game can be driven to a terminal state with a winner.
"""

import numpy as np
import pytest
from catan_engine.record import GameRecord, replay

from catan_render.bots import supported_counts
from catan_render.session import HUMAN, GameSession

# Bot kinds usable at each seat count the renderer offers.
_FOUR_PLAYER_KINDS = sorted(k for k, c in supported_counts().items() if 4 in c)
_TWO_ONLY_KINDS = sorted(k for k, c in supported_counts().items() if 2 in c and 4 not in c)


def _drive_to_completion(sess: GameSession, seed: int = 0) -> None:
    """Random legal moves for the human seats, bots played out, until the end."""
    rng = np.random.default_rng(seed)
    for _ in range(50_000):
        if sess.terminal():
            break
        legal = sess.legal_flat()
        if legal.size == 0:
            break
        sess.apply(int(rng.choice(legal)))
        sess._run_bots()


def test_acting_seat_is_human_after_construction() -> None:
    sess = GameSession(seed=0)
    # Seat 0 opens, and by default it's a human: the game waits for its move.
    assert sess.acting_seat() == 0
    assert sess.status().your_turn
    assert not sess.status().terminal


def test_default_seats_are_human_plus_random_bots() -> None:
    sess = GameSession(seed=0)
    assert sess.status().seats == [HUMAN, "random", "random", "random"]


def test_acting_seat_is_human_after_reset() -> None:
    sess = GameSession(seed=1)
    sess.apply(int(sess.legal_flat()[0]))  # perturb
    sess.reset(seed=2)
    assert sess.acting_seat() == 0
    assert sess.status().your_turn


def test_bot_step_is_none_while_human_acting() -> None:
    sess = GameSession(seed=0)
    assert sess.acting_seat() == 0
    assert sess.bot_step() is None


def test_bot_step_plays_one_bot_move() -> None:
    # The human's two setup placements hand the turn to seat 1 (a bot); each
    # bot_step then plays exactly one move until the human acts again.
    sess = GameSession(seed=0)
    sess.apply(int(sess.legal_flat()[0]))  # setup settlement
    sess.apply(int(sess.legal_flat()[0]))  # setup road
    assert sess.acting_seat() == 1
    flat = sess.bot_step()
    assert flat is not None
    sess._run_bots()
    assert sess.terminal() or sess.acting_seat() == 0
    assert sess.bot_step() is None


def test_run_bots_is_idempotent_when_human_acting() -> None:
    # When it's already the human's turn, driving the bots changes nothing.
    sess = GameSession(seed=0)
    assert sess.acting_seat() == 0
    before = sess.legal_flat().tolist()
    sess._run_bots()
    assert sess.acting_seat() == 0
    assert sess.legal_flat().tolist() == before


def test_game_drives_to_completion() -> None:
    # The human also plays a random legal move each turn; the game must reach a
    # terminal state with a real winner within a sane step budget.
    sess = GameSession(seed=0)
    _drive_to_completion(sess)
    status = sess.status()
    assert status.terminal
    assert status.winner is not None
    assert 0 <= status.winner < 4


@pytest.mark.parametrize("kind", _FOUR_PLAYER_KINDS)
def test_bot_opponents_drive_to_completion(kind: str) -> None:
    # Every bot kind that seats four players can play a full game out.
    sess = GameSession(seed=0, seats=[HUMAN, kind, kind, kind])
    assert sess.status().seats == [HUMAN, kind, kind, kind]
    _drive_to_completion(sess)
    status = sess.status()
    assert status.terminal
    assert status.winner is not None


@pytest.mark.parametrize("kind", _TWO_ONLY_KINDS)
def test_two_player_bot_opponent_drives_to_completion(kind: str) -> None:
    # The two-player-only search agents can play a full two-player game out.
    sess = GameSession(seed=0, n_players=2, seats=[HUMAN, kind])
    _drive_to_completion(sess)
    status = sess.status()
    assert status.terminal
    assert status.winner is not None


def test_all_bot_game_plays_itself() -> None:
    # No human seat: the game starts waiting on a bot (never your_turn) and
    # bot steps alone drive it to a terminal state with a winner.
    sess = GameSession(seed=0, seats=["random", "random", "random", "random"])
    assert not sess.status().your_turn
    assert sess.bot_step() is not None  # seat 0 (a bot) is acting from move one
    sess._run_bots()
    status = sess.status()
    assert status.terminal
    assert status.winner is not None


def test_human_opponents_do_not_auto_play() -> None:
    # Hotseat: with every seat human, bot_step never plays -- after each move
    # it is still some human's turn (until the game ends).
    sess = GameSession(seed=0, n_players=2, seats=[HUMAN, HUMAN])
    assert sess.status().seats == [HUMAN, HUMAN]
    rng = np.random.default_rng(0)
    seats_seen = set()
    for _ in range(200):
        if sess.terminal():
            break
        seats_seen.add(sess.acting_seat())
        assert sess.status().your_turn
        sess.apply(int(rng.choice(sess.legal_flat())))
        sess._run_bots()  # no-op: every seat is human
    assert seats_seen == {0, 1}  # both humans got turns


def test_mixed_human_and_bot_seats() -> None:
    # Seats: human, human, bot, bot. Bot play must stop on seat 1 (a human)
    # and never leave a bot seat acting.
    sess = GameSession(seed=0, seats=[HUMAN, HUMAN, "random", "random"])
    rng = np.random.default_rng(0)
    for _ in range(300):
        if sess.terminal():
            break
        assert sess.acting_seat() in (0, 1)
        sess.apply(int(rng.choice(sess.legal_flat())))
        sess._run_bots()


def test_two_player_session() -> None:
    # n_players=2 seats the human and one bot, and a random game still drives
    # to completion with a seated winner.
    sess = GameSession(seed=0, n_players=2)
    assert sess.acting_seat() == 0
    assert len(sess.board[1].player_resources[0]) == 2  # player axis = seats
    _drive_to_completion(sess)
    status = sess.status()
    assert status.terminal
    assert status.winner is not None and 0 <= status.winner < 2


def test_reset_keeps_seat_count_unless_changed() -> None:
    sess = GameSession(seed=0, n_players=2)
    sess.reset(seed=1)  # no n_players -> keeps 2
    assert sess.n_players == 2
    sess.reset(seed=2, n_players=4)
    assert sess.n_players == 4


def test_invalid_seats_rejected() -> None:
    with pytest.raises(ValueError, match="unknown seat"):
        GameSession(seed=0, seats=[HUMAN, "clever", "random", "random"])
    with pytest.raises(ValueError, match="expected 4 seats"):
        GameSession(seed=0, seats=[HUMAN, "random"])


@pytest.mark.skipif(not _TWO_ONLY_KINDS, reason="no two-player-only bot kinds")
def test_seat_kind_must_support_player_count() -> None:
    # A two-player-only agent can't seat a four-player game.
    with pytest.raises(ValueError, match="not available in a 4-player game"):
        GameSession(seed=0, seats=[HUMAN, _TWO_ONLY_KINDS[0], "random", "random"])


def test_log_records_moves_and_chat() -> None:
    sess = GameSession(seed=0)
    assert sess.log() == []
    sess.apply(int(sess.legal_flat()[0]))  # human setup settlement
    (entry,) = sess.log()
    assert entry.kind == "move"
    assert entry.player == 0
    assert entry.action_type == "setup_settlement"
    sess.apply(int(sess.legal_flat()[0]))  # human setup road
    sess._run_bots()
    assert len(sess.log()) > 2  # the bots' moves were logged too
    assert all(e.kind == "move" for e in sess.log())
    sess.add_chat(0, "hello")
    assert sess.log()[-1].kind == "chat" and sess.log()[-1].text == "hello"
    sess.add_chat(None, "just watching")  # a spectator line
    assert sess.log()[-1].player is None
    sess.reset(seed=1)
    assert sess.log() == []


def test_log_rolls_and_win() -> None:
    sess = GameSession(seed=0)
    _drive_to_completion(sess)
    entries = sess.log()
    # Rolls carry the rolled value...
    rolls = [e for e in entries if e.action_type == "roll_dice"]
    assert rolls and all(e.text.startswith("rolled ") for e in rolls)
    # ...and the game's end is logged exactly once, for the winning seat.
    wins = [e for e in entries if e.kind == "win"]
    assert len(wins) == 1
    assert wins[0].player == sess.status().winner


def test_record_exports_a_replayable_game() -> None:
    sess = GameSession(seed=3, n_players=2, seats=["random", "random"])
    sess._run_bots()  # all-bot game plays itself out
    rec = sess.record()
    assert rec.winner == sess.status().winner
    assert rec.meta == {"seats": ["random", "random"]}
    assert len(rec.moves) > 10  # the full trace, not the capped log
    # The JSON roundtrips and the engine replays it without complaint.
    rec2 = GameRecord.from_json(rec.to_json())
    boards = list(replay(rec2))
    assert len(boards) == len(rec.moves)


def test_record_of_running_game_has_no_winner() -> None:
    sess = GameSession(seed=0)
    sess.apply(int(sess.legal_flat()[0]))
    rec = sess.record()
    assert rec.winner is None
    assert [m.flat for m in rec.moves] == [rec.moves[0].flat]
    sess.reset(seed=1)
    assert sess.record().moves == ()  # reset starts a fresh trace
