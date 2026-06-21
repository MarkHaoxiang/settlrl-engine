// Hook driving one lobby: the current snapshot over its SSE stream, plus chat.
// The server pushes a fresh per-seat snapshot on every change (joins, seat
// edits, config, the start signal), so nothing is polled. A dropped stream
// reconnects with a short backoff; a 404 means the lobby closed.

import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./api";
import { chatLobby, streamLobby, type LobbySnapshot } from "./lobby";
import type { SeatTokens } from "./seats";

const RECONNECT_MS = 1000;

export interface UseLobby {
  snapshot: LobbySnapshot | null;
  error: string | null;
  chat: (text: string, player: number | null) => void;
}

export function useLobby(lobbyId: string | null, tokens: SeatTokens): UseLobby {
  const [snapshot, setSnapshot] = useState<LobbySnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSnapshot(null);
    setError(null);
    if (!lobbyId) return;
    const controller = new AbortController();
    let stopped = false;
    void (async () => {
      while (!stopped) {
        try {
          for await (const snap of streamLobby(lobbyId, tokens, controller.signal)) {
            if (!stopped) setSnapshot(snap);
          }
        } catch (e) {
          if (stopped) return;
          if (e instanceof ApiError && e.status === 404) {
            setError("This lobby has closed.");
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
  }, [lobbyId, tokens]);

  const chat = useCallback(
    (text: string, player: number | null) => {
      if (lobbyId) void chatLobby(lobbyId, tokens, text, player).then(setSnapshot, () => {});
    },
    [lobbyId, tokens]
  );

  return { snapshot, error, chat };
}
