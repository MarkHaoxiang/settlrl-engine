// Hook driving the live game: the current snapshot plus act / reset.
//
// Every call replaces the snapshot with the server's response. Bot seats are
// played one move per request (`postBotStep`): whenever the snapshot says a bot
// is acting, the hook waits BOT_STEP_DELAY_MS and steps it — so each bot move
// lands as its own snapshot and the board can animate it. `busy` is true while
// a request is in flight; a 409 (the move stopped being legal) refreshes the
// snapshot instead of erroring.

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import {
  fetchGame,
  postAction,
  postBotStep,
  postChat,
  postReset,
  type GameSnapshot,
  type NewGameConfig,
} from "./game";

// Pause between bot moves, so each one registers visually.
const BOT_STEP_DELAY_MS = 650;

export interface UseGame {
  snapshot: GameSnapshot | null;
  error: string | null;
  busy: boolean;
  act: (flat: number) => void;
  reset: (config: NewGameConfig) => void;
  chat: (text: string, player: number | null) => void;
}

export function useGame(): UseGame {
  const [snapshot, setSnapshot] = useState<GameSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Set when a bot step made no progress (a stuck game); stops the bot loop
  // until the human acts or resets.
  const stalled = useRef(false);

  const run = useCallback(async (op: () => Promise<GameSnapshot>) => {
    setBusy(true);
    try {
      setSnapshot(await op());
      setError(null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setSnapshot(await fetchGame());
      else setError(`Game request failed: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void run(fetchGame);
  }, [run]);

  // While a bot seat is acting, step one bot move after a short pause.
  useEffect(() => {
    if (!snapshot || snapshot.status.your_turn || snapshot.status.terminal) return;
    if (stalled.current) return;
    const timer = setTimeout(
      () =>
        void run(async () => {
          const next = await postBotStep();
          stalled.current = next.bot_move === null && !next.status.your_turn && !next.status.terminal;
          return next;
        }),
      BOT_STEP_DELAY_MS
    );
    return () => clearTimeout(timer);
  }, [snapshot, run]);

  return {
    snapshot,
    error,
    busy,
    act: useCallback(
      (flat) => {
        stalled.current = false;
        void run(() => postAction(flat));
      },
      [run]
    ),
    reset: useCallback(
      (config) => {
        stalled.current = false;
        void run(() => postReset(config));
      },
      [run]
    ),
    chat: useCallback(
      (text, player) => {
        void run(() => postChat(text, player));
      },
      [run]
    ),
  };
}
