// React Query hooks for the read-only REST endpoints, over the typed client.
// These replace the per-component fetch + useState + useEffect dance with cached
// queries (loading/error state, refetch, dedupe) and schema-derived types.

import { useQuery } from "@tanstack/react-query";

import type { AuthUser } from "./auth";
import { client, unwrap } from "./client";
import type { components } from "./api-schema";

type Schemas = components["schemas"];
export type MyGame = Schemas["_MyGameModel"];
export type PastGame = Schemas["_PastGameModel"];
export type LeaderboardEntry = Schemas["_LeaderboardEntry"];

// The signed-in user's in-progress games (seats follow the account); idle until
// signed in.
export function useMyGames(user: AuthUser | null) {
  return useQuery({
    queryKey: ["me", "games"],
    queryFn: () => unwrap(client.GET("/api/me/games")),
    enabled: !!user,
  });
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
