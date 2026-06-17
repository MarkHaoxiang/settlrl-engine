// Hook driving one live game: the current snapshot plus act / chat.
//
// The snapshot arrives over the game's SSE stream — the server pushes a new
// per-seat view on every state change (moves, server-paced bot plays, chat,
// joins), so nothing is polled. POST responses also carry a snapshot for
// immediate feedback; versions order the two sources, late arrivals are
// dropped. A dropped stream reconnects with a short backoff; `busy` is true
// while a request is in flight; a 409 (the move stopped being legal) is
// ignored — the stream already delivered the snapshot that outdated it.

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import { postAction, postChat, streamGame, type GameSnapshot } from "./game";
import type { SeatTokens } from "./seats";
import { play, soundForLog } from "./sound";

const RECONNECT_MS = 1000;

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
  const version = useRef(-1);
  // Sound is played off newly-applied log entries; primed on the first snapshot
  // so a backlog (reconnect / mid-game join) isn't replayed all at once.
  const primed = useRef(false);
  const lastSoundId = useRef(-1);

  const show = useCallback((snap: GameSnapshot) => {
    if (snap.version <= version.current) return;
    version.current = snap.version;
    if (primed.current) {
      for (const entry of snap.log) {
        if (entry.id <= lastSoundId.current) continue;
        const sound = soundForLog(entry);
        if (sound) void play(sound);
      }
    }
    primed.current = true;
    if (snap.log.length) lastSoundId.current = snap.log[snap.log.length - 1].id;
    setSnapshot(snap);
  }, []);

  useEffect(() => {
    version.current = -1;
    primed.current = false;
    lastSoundId.current = -1;
    setSnapshot(null);
    setError(null);
    if (!gameId) return;
    const controller = new AbortController();
    let stopped = false;
    void (async () => {
      while (!stopped) {
        try {
          for await (const snap of streamGame(gameId, tokens, controller.signal)) show(snap);
        } catch (e) {
          if (stopped) return;
          if (e instanceof ApiError && e.status === 404) {
            setError("No such game — it may have ended and been cleaned up.");
            return;
          }
        }
        if (!stopped) await new Promise((r) => setTimeout(r, RECONNECT_MS));
      }
    })();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [gameId, tokens, show]);

  const run = useCallback(
    async (op: () => Promise<GameSnapshot>) => {
      setBusy(true);
      try {
        show(await op());
        setError(null);
      } catch (e) {
        // A 409 means the move stopped being legal (a race) — give the audible
        // nudge but no error banner; the stream already delivered the snapshot
        // that outdated it.
        if (e instanceof ApiError && e.status === 409) void play("error");
        else setError(`Game request failed: ${String(e)}`);
      } finally {
        setBusy(false);
      }
    },
    [show]
  );

  return {
    snapshot,
    error,
    busy,
    act: useCallback(
      (flat) => {
        if (gameId) void run(() => postAction(gameId, tokens, flat));
      },
      [gameId, tokens, run]
    ),
    chat: useCallback(
      (text, player) => {
        if (gameId) void run(() => postChat(gameId, tokens, text, player));
      },
      [gameId, tokens, run]
    ),
  };
}
