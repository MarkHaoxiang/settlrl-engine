"""End-to-end differential test: the engine and the ``catan-reference`` oracle
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

import random

import catan_reference as ref
import jax.numpy as jnp
import numpy as np
from catan_engine.board import Board, make_board
from catan_engine.board.state import BoardState
from catan_engine.env import step
from catan_engine.mechanics.action import ActionParams, ActionResult

from tests import conversion as conv

_INVALID = int(ActionResult.INVALID)
_GAME_COMPLETE = int(ActionResult.GAME_COMPLETE)


def _params(idx: int, target: int) -> ActionParams:
    return ActionParams(
        idx=jnp.asarray([idx], jnp.int32),
        target=jnp.asarray([target], jnp.int32),
    )


def _inject_outcome(
    action: ref.Action, old: Board, new_state: object, current_player: int
) -> ref.Action:
    """Fill a reference action's stochastic field from the engine's realised step."""
    if isinstance(action, ref.Roll):
        return ref.Roll(value=int(np.asarray(new_state.dice_roll[0])))  # type: ignore[attr-defined]
    if isinstance(action, ref.BuyDevelopmentCard):
        before = np.asarray(old[1].dev_hand[0, current_player]).astype(int)
        after = np.asarray(new_state.dev_hand[0, current_player]).astype(int)  # type: ignore[attr-defined]
        drawn = int(np.argmax(after - before))
        return ref.BuyDevelopmentCard(card=ref.DevCard(drawn))
    if (
        isinstance(action, (ref.MoveRobber, ref.PlayKnight))
        and action.victim is not None
    ):
        before = np.asarray(old[1].player_resources[0, current_player]).astype(int)
        after = np.asarray(new_state.player_resources[0, current_player]).astype(int)  # type: ignore[attr-defined]
        stolen = ref.Resource(int(np.argmax(after - before)))
        # dataclasses.replace would also work; reconstruct to keep mypy simple.
        return type(action)(action.tile, action.victim, stolen)
    return action


def _try_engine(board: Board, action: ref.Action) -> tuple[BoardState, int]:
    """Apply one reference action to the engine; ``(new_state, result code)``."""
    atype, idx, target = conv.to_engine_action(action)
    new_state, code = step(board, jnp.asarray([atype], jnp.int32), _params(idx, target))
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


def _play_one_game(
    seed: int, max_steps: int = 300, n_players: int = 4, bundles: bool = False
) -> None:
    board = make_board(1, seed=seed, n_players=n_players)
    game = conv.to_reference_single(board, 0)
    rng = random.Random(seed)

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
        if bundles and game.phase is ref.Phase.MAIN and game.has_rolled:
            # The flat table (and so the cross-check above) only names 1:1
            # proposals: probe the packed bundle domain directly, both ways.
            for _probe in range(3):
                probe = _random_bundle(rng, game)
                _, code = _try_engine(board, probe)
                assert (code != _INVALID) == game.is_legal(probe), (
                    f"seed={seed}: bundle legality mismatch on {probe!r}: "
                    f"engine={code != _INVALID} reference={game.is_legal(probe)}"
                )
            if rng.random() < 0.25 and (bundle := _draw_legal_bundle(rng, game)):
                new_state, result = _try_engine(board, bundle)
                assert result != _INVALID, (
                    f"seed={seed}: engine rejected legal bundle {bundle!r}"
                )
                applied = bundle
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
    from catan_engine.board import to_main
    from catan_engine.board.dev_cards import DevCard

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


def test_bundle_trades_match_reference_over_random_games() -> None:
    # Multi-card bundle coverage: probes random (mostly illegal) bundles every
    # MAIN step and plays injected legal ones through their accept / reject.
    for n_players, seed in ((3, 5), (4, 6)):
        _play_one_game(seed, n_players=n_players, bundles=True)
