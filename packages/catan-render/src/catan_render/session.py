"""A single live Catan game, driven through the engine's AEC wrapper.

The renderer plays one game at a time against random bots. ``GameSession`` wraps
``catan_engine.env.aec.CatanAECEnv`` (a single-game PettingZoo-AEC env): the human is
seat 0, and after each human move the session auto-plays *random legal* moves for
the other seats until it is the human's turn again or the game ends.

The session exposes just what the server needs: the underlying engine board (for
``convert.board_to_model``), the legal flat-action indices for the acting player,
and a small status snapshot (phase / dice / whose turn / winner).
"""

from __future__ import annotations

import numpy as np
from catan_engine.env.aec import CatanAECEnv
from catan_engine.board import Board
from catan_engine.board.state import VICTORY_POINTS_TO_WIN, GamePhase

from .models import GameStatusModel

# The human always plays seat 0; every other seat is auto-played by a bot.
HUMAN_SEAT = 0

# Guard against a pathological non-terminating bot loop (a full game is well
# under this many engine steps).
_MAX_BOT_STEPS = 50_000


class IllegalActionError(ValueError):
    """Raised when an action that is not currently legal is applied."""


class GameSession:
    """A live game vs. random bots, behind the single-game AEC env.

    ``n_players`` (2..4) is how many seats the game has: the human plus
    ``n_players - 1`` bots.
    """

    def __init__(self, seed: int = 0, n_players: int = 4) -> None:
        self.n_players = n_players
        self.reset(seed)

    def reset(self, seed: int = 0, n_players: int | None = None) -> None:
        """Start a fresh game (``n_players`` changes the seat count; None keeps it)."""
        if n_players is not None:
            self.n_players = n_players
        self.seed = seed
        self.env = CatanAECEnv(seed=seed, n_players=self.n_players)
        # Dedicated RNG so bot choices are reproducible per seed and independent
        # of the engine's own randomness.
        self._rng = np.random.default_rng(seed)
        # If the opening seat is somehow a bot, let it play up to the human.
        self._run_bots()

    # -- engine views -----------------------------------------------------

    @property
    def board(self) -> Board:
        """The underlying batched ``(BoardLayout, BoardState)`` (one game)."""
        return self.env._env.board

    def acting_seat(self) -> int:
        return int(self.env._env.agent_selection[0])

    def terminal(self) -> bool:
        return all(self.env.terminations.values())

    def legal_flat(self) -> np.ndarray:
        """Flat indices of the actions legal for the acting player right now."""
        mask = np.asarray(self.env.observe(self.env.agent_selection)["action_mask"])
        return np.flatnonzero(mask)

    # -- moves ------------------------------------------------------------

    def apply(self, flat: int) -> None:
        """Apply the human's chosen flat action, then let the bots respond."""
        if flat not in self.legal_flat():
            raise IllegalActionError(f"action {flat} is not legal right now")
        self.env.step(int(flat))
        self._run_bots()

    def _run_bots(self) -> None:
        """Play random legal moves while it is a bot's turn (and not game over)."""
        steps = 0
        while not self.terminal() and self.acting_seat() != HUMAN_SEAT:
            legal = self.legal_flat()
            if legal.size == 0:
                break
            self.env.step(int(self._rng.choice(legal)))
            steps += 1
            if steps >= _MAX_BOT_STEPS:
                break

    # -- status -----------------------------------------------------------

    def status(self) -> GameStatusModel:
        """A snapshot of turn flow for the wire model."""
        state = self.env._env._state
        vps = np.asarray(self.env._env._vps[0])
        terminal = self.terminal()
        winner = (
            int(np.argmax(vps))
            if terminal and bool((vps >= VICTORY_POINTS_TO_WIN).any())
            else None
        )
        acting = self.acting_seat()
        return GameStatusModel(
            phase=GamePhase(int(state.phase[0])).name.lower(),
            current_player=int(state.current_player[0]),
            acting_player=acting,
            dice_roll=int(state.dice_roll[0]),
            has_rolled=bool(state.has_rolled[0]),
            your_turn=(not terminal) and acting == HUMAN_SEAT,
            terminal=terminal,
            winner=winner,
        )
