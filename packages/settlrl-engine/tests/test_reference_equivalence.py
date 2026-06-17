"""End-to-end differential test: the engine and the ``settlrl-reference`` oracle
must agree on the *whole* game state, step after step.

For each seed we build an engine board (batch size 1) and the matching reference
``Game`` (via ``tests.conversion``), then play a capped random game. Each step we
take the reference's legal actions (the gold standard), drive the engine with one
the engine also accepts, recover the engine's realised random outcome (the dice
roll, the drawn development card, the stolen resource) by diffing its state, feed
that same outcome into the reference, and assert the two states are identical.

This exercises setup, rolling, the 7 (discard + robber + steal), building, the
maritime trade, every development card, the awards, and winning -- far more than
the per-rule unit tests in ``mechanics/``.

Two fuzz layers close the directions a reference-driven stream cannot see:

- every step, ``assert_legality_match`` compares the engine's flat mask against
  reference ``is_legal`` over the whole flat table, so an engine that is more
  (or differently) permissive than the reference fails even though the stream
  never plays the disputed move;
- with ``bundles=True``, random multi-card trade proposals are probed against
  both engines (the flat table only names 1:1 trades) and reference-legal ones
  are injected into the stream and played through their accept / reject.
"""

from __future__ import annotations

import copy
import random

import jax
import jax.numpy as jnp
import numpy as np
import settlrl_reference as ref
from settlrl_engine.belief import BeliefState, make_belief, update_belief
from settlrl_engine.board import Board, make_board, to_main
from settlrl_engine.board.dev_cards import DevCard
from settlrl_engine.board.state import BoardState
from settlrl_engine.env import step
from settlrl_engine.mechanics.action import ActionParams, ActionResult

from tests import conversion as conv

_INVALID = int(ActionResult.INVALID)
_GAME_COMPLETE = int(ActionResult.GAME_COMPLETE)
_NR = len(ref.RESOURCES)

# The engine's per-observer card counting, single-game (the env vmaps it over a
# batch); jitted so each step reuses one trace.
_update_belief = jax.jit(update_belief)


def _assert_belief_matches(engine: BeliefState, oracle: ref.Belief, seed: int) -> None:
    """The engine's belief bounds equal the reference oracle's, entry for entry.

    Indices align directly: observers and players share an order, engine
    resource column ``r`` is ``Resource(r)``, dev-card slot ``c`` is
    ``DevCard(c)``.
    """
    lo = np.asarray(engine.res_lo).astype(int)  # (observers, players, resources)
    hi = np.asarray(engine.res_hi).astype(int)
    played = np.asarray(engine.dev_played).astype(int)
    n = oracle.n_players
    for o in range(n):
        for p in range(n):
            for r in range(_NR):
                assert lo[o][p][r] == oracle.res_lo[o][p][r], (
                    f"seed={seed}: res_lo[{o}][{p}][{r}] engine={lo[o][p][r]} "
                    f"reference={oracle.res_lo[o][p][r]}"
                )
                assert hi[o][p][r] == oracle.res_hi[o][p][r], (
                    f"seed={seed}: res_hi[{o}][{p}][{r}] engine={hi[o][p][r]} "
                    f"reference={oracle.res_hi[o][p][r]}"
                )
    for card, count in oracle.dev_played.items():
        assert played[int(card)] == count, (
            f"seed={seed}: dev_played[{card}] engine={played[int(card)]} "
            f"reference={count}"
        )


def _inject_outcome(
    action: ref.Action, old: Board, new_state: BoardState, current_player: int
) -> ref.Action:
    """Fill a reference action's stochastic field from the engine's realised step."""
    if isinstance(action, ref.Roll):
        return ref.Roll(value=int(np.asarray(new_state.dice_roll[0])))
    if isinstance(action, ref.BuyDevelopmentCard):
        before = np.asarray(old[1].dev_hand[0, current_player]).astype(int)
        after = np.asarray(new_state.dev_hand[0, current_player]).astype(int)
        return ref.BuyDevelopmentCard(card=ref.DevCard(int(np.argmax(after - before))))
    if (
        isinstance(action, (ref.MoveRobber, ref.PlayKnight))
        and action.victim is not None
    ):
        before = np.asarray(old[1].player_resources[0, current_player]).astype(int)
        after = np.asarray(new_state.player_resources[0, current_player]).astype(int)
        stolen = ref.Resource(int(np.argmax(after - before)))
        # dataclasses.replace would also work; reconstruct to keep mypy simple.
        return type(action)(action.tile, action.victim, stolen)
    return action


def _try_engine(board: Board, action: ref.Action) -> tuple[BoardState, int]:
    """Apply one reference action to the engine; ``(new_state, result code)``."""
    atype, idx, target = conv.to_engine_action(action)
    params = ActionParams(
        idx=jnp.asarray([idx], jnp.int32), target=jnp.asarray([target], jnp.int32)
    )
    new_state, code = step(board, jnp.asarray([atype], jnp.int32), params)
    return new_state, int(np.asarray(code[0]))


def _random_bundle(rng: random.Random, game: ref.Game) -> ref.ProposeTrade:
    """An arbitrary bundle proposal -- often illegal (overlapping sides, empty
    sides, unaffordable, partner == proposer), to probe the rejection paths."""
    return ref.ProposeTrade(
        partner=rng.randrange(game.n_players),
        give=tuple(rng.randint(0, 2) for _ in ref.RESOURCES),
        receive=tuple(rng.randint(0, 2) for _ in ref.RESOURCES),
    )


def _draw_legal_bundle(rng: random.Random, game: ref.Game) -> ref.ProposeTrade | None:
    """A random reference-legal bundle proposal from the current hand, or None."""
    p = game.current_player
    hand = game.players[p].resources
    held = [r for r in ref.RESOURCES if hand[r] > 0]
    if not held:
        return None
    giving = rng.sample(held, k=rng.randint(1, min(2, len(held))))
    asking = rng.sample(
        [r for r in ref.RESOURCES if r not in giving], k=rng.randint(1, 2)
    )
    proposal = ref.ProposeTrade(
        partner=rng.choice([q for q in range(game.n_players) if q != p]),
        give=tuple(
            rng.randint(1, min(3, hand[r])) if r in giving else 0 for r in ref.RESOURCES
        ),
        receive=tuple(rng.randint(1, 3) if r in asking else 0 for r in ref.RESOURCES),
    )
    return proposal if game.is_legal(proposal) else None


def _fuzz_bundles(
    rng: random.Random, board: Board, game: ref.Game, seed: int
) -> tuple[ref.ProposeTrade, BoardState, int] | None:
    """Probe the packed bundle domain, which the flat table (and so the
    per-step legality cross-check) only samples at 1:1: arbitrary bundles are
    checked against both engines, and sometimes a legal one is dealt for
    injection into the stream (returned with its applied engine step)."""
    for _ in range(3):
        probe = _random_bundle(rng, game)
        _, code = _try_engine(board, probe)
        legal = game.is_legal(probe)
        assert (code != _INVALID) == legal, (
            f"seed={seed}: bundle legality mismatch on {probe!r}: "
            f"engine={code != _INVALID} reference={legal}"
        )
    if rng.random() < 0.25 and (bundle := _draw_legal_bundle(rng, game)):
        state, result = _try_engine(board, bundle)
        assert result != _INVALID, f"seed={seed}: engine rejected {bundle!r}"
        return bundle, state, result
    return None


def _play_one_game(
    seed: int,
    max_steps: int = 300,
    n_players: int = 4,
    bundles: bool = False,
    track_belief: bool = False,
) -> None:
    board = make_board(1, seed=seed, n_players=n_players)
    game = conv.to_reference_single(board, 0)
    rng = random.Random(seed)
    # Card counting, advanced in lockstep when requested: the engine's tracker
    # (single-game slice) against the reference oracle.
    engine_belief = jax.tree.map(lambda x: x[0], make_belief(1, n_players))
    oracle_belief = ref.Belief.new(n_players)

    for _ in range(max_steps):
        # Only GAME_OVER has no legal actions, so this doubles as the terminal
        # check. (Avoids narrowing ``game.phase`` on an identity test, which the
        # type checker wouldn't reset across the mutating ``game.apply`` below.)
        legal = game.legal_actions()
        if not legal:
            return
        rng.shuffle(legal)
        conv.assert_legality_match(board, game)

        applied: ref.Action | None = None
        result = _INVALID
        if (
            bundles
            and game.phase is ref.Phase.MAIN
            and game.has_rolled
            and (injected := _fuzz_bundles(rng, board, game, seed))
        ):
            applied, new_state, result = injected
        if applied is None:
            for candidate in legal:
                new_state, result = _try_engine(board, candidate)
                if result != _INVALID:
                    applied = candidate
                    break
        assert applied is not None, (
            f"seed={seed}: engine rejected every reference-legal action in "
            f"phase {game.phase}"
        )

        current_player = int(np.asarray(board[1].current_player[0]))
        ref_action = _inject_outcome(applied, board, new_state, current_player)

        if track_belief:
            atype, idx, target = conv.to_engine_action(ref_action)
            params = ActionParams(
                idx=jnp.asarray(idx, jnp.int32), target=jnp.asarray(target, jnp.int32)
            )
            before = copy.deepcopy(game)
            engine_belief = _update_belief(
                engine_belief,
                jax.tree.map(lambda x: x[0], board[1]),
                jax.tree.map(lambda x: x[0], new_state),
                jnp.asarray(atype, jnp.int32),
                params,
            )
            game.apply(ref_action)
            oracle_belief.update(before, game, ref_action)
            _assert_belief_matches(engine_belief, oracle_belief, seed)
        else:
            game.apply(ref_action)
        board = (board[0], new_state)

        if result == _GAME_COMPLETE:
            # The engine signals the win via the result code (phase unchanged);
            # the reference moves to GAME_OVER. Check they agree and stop.
            assert game.phase is ref.Phase.GAME_OVER, (
                f"seed={seed}: engine reported a win the reference did not "
                f"({ref_action})"
            )
            conv.assert_states_match(board, game, 0, ignore_phase=True)
            return
        conv.assert_states_match(board, game, 0)


# A small spread of seeds; seed 18 reaches a win (Knight -> Largest Army -> 10
# VP), so it covers the game-completion path as well as the open-game state
# agreement. Kept short on purpose -- each game is a full step-by-step
# differential replay, so this test is one of the slowest in the suite.
_SEEDS = (0, 1, 2, 18)


def test_engine_matches_reference_over_random_games() -> None:
    for seed in _SEEDS:
        _play_one_game(seed)


def test_engine_matches_reference_with_fewer_players() -> None:
    # 2- and 3-player games exercise the shorter setup snake, the tighter
    # end-turn rotation, and the unused player rows staying empty. Fewer seeds
    # than the 4-player run to keep the suite's slowest test in check.
    for n_players in (2, 3):
        for seed in (0, 1):
            _play_one_game(seed, n_players=n_players)


def test_turn_start_win_claim_matches_reference() -> None:
    # Rulebook p.5: a player at 10+ VP out of turn wins at the *start of their
    # own turn*, not immediately. Hand player 1 ten VP of dev cards mid player
    # 0's turn and walk the END_TURN claim through both engines.
    layout, st = to_main(make_board(1, seed=2, n_players=3))
    st = st._replace(
        dev_hand=st.dev_hand.at[0, 1, DevCard.VICTORY_POINT].set(10),
    )
    board: Board = (layout, st)
    game = conv.to_reference_single(board, 0)
    # Mid player 0's turn nothing has ended: both engines still agree on a
    # full set of legal moves.
    conv.assert_legality_match(board, game)

    new_state, result = _try_engine(board, ref.EndTurn())
    game.apply(ref.EndTurn())
    assert result == _GAME_COMPLETE, "engine must complete on the turn-start claim"
    assert game.phase is ref.Phase.GAME_OVER
    assert game.current_player == 1
    conv.assert_states_match((board[0], new_state), game, 0, ignore_phase=True)


def test_belief_bounds_match_reference_over_random_games() -> None:
    # The engine's card counting (settlrl_engine.belief) is itself differentially
    # checked: its per-observer bounds and public dev tally must equal the
    # reference oracle's at every step. Seed 18 reaches a win via a Knight, so it
    # exercises hidden third-party steals at 4p; a 2-player seed covers the
    # exact-bounds case (every flow is mutually visible).
    _play_one_game(18, n_players=4, track_belief=True)
    _play_one_game(0, n_players=2, track_belief=True)


def test_bundle_trades_match_reference_over_random_games() -> None:
    # Multi-card bundle coverage: probes random (mostly illegal) bundles every
    # MAIN step and plays injected legal ones through their accept / reject.
    for n_players, seed in ((3, 5), (4, 6)):
        _play_one_game(seed, n_players=n_players, bundles=True)
