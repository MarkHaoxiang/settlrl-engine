// React Query hooks for the read-only REST endpoints, over the typed client.
// These replace the per-component fetch + useState + useEffect dance with cached
// queries (loading/error state, refetch, dedupe) and schema-derived types.

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import type { AuthUser } from "./auth";
import { client, unwrap } from "./client";
import { clearCurrentGame, currentGameId } from "./seats";
import type { components } from "./api-schema";

type Schemas = components["schemas"];
export type MyGame = Schemas["_MyGameModel"];
export type PastGame = Schemas["_PastGameModel"];
export type LeaderboardEntry = Schemas["_LeaderboardEntry"];
export type LobbyGame = Schemas["_LobbyGameModel"];

// The signed-in user's in-progress games (seats follow the account); idle until
// signed in.
export function useMyGames(user: AuthUser | null) {
  return useQuery({
    queryKey: ["me", "games"],
    queryFn: () => unwrap(client.GET("/api/me/games")),
    enabled: !!user,
  });
}

// The one live game the user is currently in, or null — drives the "you're
// already in a game" gate. Signed in: the account's first live game (server
// truth). Guest: the locally-tracked game, verified live and forgotten once it's
// gone or finished. Returns the id optimistically while the guest probe loads,
// so the gate never briefly lets a second game through.
export function useCurrentGame(user: AuthUser | null): string | null {
  const mine = useMyGames(user);
  const localId = currentGameId();
  const guest = useQuery({
    queryKey: ["game-live", localId],
    queryFn: () =>
      unwrap(
        client.GET("/api/games/{game_id}", { params: { path: { game_id: localId! } } })
      ),
    enabled: !user && !!localId,
    retry: false,
  });
  const guestDead =
    !user && !!localId && (guest.isError || (guest.data?.status.terminal ?? false));
  useEffect(() => {
    if (guestDead && localId) clearCurrentGame(localId);
  }, [guestDead, localId]);

  if (user) return mine.data?.[0]?.id ?? null;
  return localId && !guestDead ? localId : null;
}

// The signed-in user's finished games, newest first; idle until signed in.
export function useHistory(user: AuthUser | null) {
  return useQuery({
    queryKey: ["me", "history"],
    queryFn: () => unwrap(client.GET("/api/me/history")),
    enabled: !!user,
  });
}

// The public per-player-count Elo ladders.
export function useLeaderboard() {
  return useQuery({
    queryKey: ["leaderboard"],
    queryFn: () => unwrap(client.GET("/api/leaderboard")),
  });
}

// Open games anyone can join, newest first; polled so the list stays live.
export function useLobby() {
  return useQuery({
    queryKey: ["lobby"],
    queryFn: () => unwrap(client.GET("/api/lobby")),
    refetchInterval: 4000,
  });
}
