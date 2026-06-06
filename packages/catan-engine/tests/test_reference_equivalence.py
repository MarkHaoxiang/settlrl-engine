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
"""

from __future__ import annotations

import random

import jax.numpy as jnp
import numpy as np

import catan_reference as ref
from catan_engine.board import Board, make_board
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


def _play_one_game(seed: int, max_steps: int = 300) -> None:
    board = make_board(1, seed=seed)
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

        applied: ref.Action | None = None
        result = _INVALID
        for candidate in legal:
            atype, idx, target = conv.to_engine_action(candidate)
            new_state, code = step(
                board, jnp.asarray([atype], jnp.int32), _params(idx, target)
            )
            result = int(np.asarray(code[0]))
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
