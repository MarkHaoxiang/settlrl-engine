// Hook driving one live game: the current snapshot plus act / chat.
//
// Every call replaces the snapshot with the server's response, requested with
// this client's seat tokens (so the view is per-seat). Bot seats are played
// one move per request (`postBotStep`): whenever the snapshot says a bot is
// acting, the hook waits BOT_STEP_DELAY_MS and steps it — so each bot move
// lands as its own snapshot and the board can animate it. `busy` is true
// while a request is in flight; a 409 (the move stopped being legal)
// refreshes the snapshot instead of erroring.

import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./api";
import { fetchGame, postAction, postBotStep, postChat, type GameSnapshot } from "./game";
import type { SeatTokens } from "./seats";

// Pause between bot moves, so each one registers visually; the slower poll
// runs while another human is thinking (no bot move was due last time).
const BOT_STEP_DELAY_MS = 650;
const WAIT_POLL_MS = 1500;

export interface UseGame {
  snapshot: GameSnapshot | null;
  error: string | null;
  busy: boolean;
  act: (flat: number) => void;
  chat: (text: string, player: number | null) => void;
}

export function useGame(gameId: string | null, tokens: SeatTokens): UseGame {
  const [snapshot, setSnapshot] = useState<GameSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = useCallback(
    async (op: () => Promise<GameSnapshot>) => {
      if (!gameId) return;
      setBusy(true);
      try {
        setSnapshot(await op());
        setError(null);
      } catch (e) {
        if (e instanceof ApiError && e.status === 409) setSnapshot(await fetchGame(gameId, tokens));
        else setError(`Game request failed: ${String(e)}`);
      } finally {
        setBusy(false);
      }
    },
    [gameId, tokens]
  );

  useEffect(() => {
    setSnapshot(null);
    if (gameId) void run(() => fetchGame(gameId, tokens));
  }, [gameId, tokens, run]);

  // While it isn't this client's turn, keep the game moving and the view
  // fresh: each poll steps one due bot move (a no-op while another human is
  // thinking) and returns the latest snapshot. Several clients polling
  // duplicates pacing harmlessly — the server's per-game lock serialises.
  useEffect(() => {
    if (!gameId || !snapshot || snapshot.status.your_turn || snapshot.status.terminal) return;
    const delay = snapshot.bot_move === null ? WAIT_POLL_MS : BOT_STEP_DELAY_MS;
    const timer = setTimeout(() => void run(() => postBotStep(gameId, tokens)), delay);
    return () => clearTimeout(timer);
  }, [gameId, tokens, snapshot, run]);

  return {
    snapshot,
    error,
    busy,
    act: useCallback(
      (flat) => {
        if (!gameId) return;
        void run(() => postAction(gameId, tokens, flat));
      },
      [gameId, tokens, run]
    ),
    chat: useCallback(
      (text, player) => {
        if (!gameId) return;
        void run(() => postChat(gameId, tokens, text, player));
      },
      [gameId, tokens, run]
    ),
  };
}
