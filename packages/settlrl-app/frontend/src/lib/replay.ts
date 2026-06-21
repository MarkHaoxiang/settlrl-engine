// Replay API client (/api/replay*): load a game record on the server and
// fetch the board snapshot + move log at any point of the game.

import { api } from "./api";
import type { components } from "./api-schema";
import { adaptBoard, type Board } from "./boardData";

type ReplayWire = components["schemas"]["ReplayStateModel"];

// The wire snapshot with its board adapted to frontend coordinates.
export interface ReplayState extends Omit<ReplayWire, "board"> {
  board: Board;
}

const adapt = (wire: ReplayWire): ReplayState => ({
  ...wire,
  board: adaptBoard(wire.board),
});

// Load a record document (the JSON from GET /api/game/record / a saved file).
export async function loadReplay(doc: unknown): Promise<ReplayState> {
  return adapt(
    await api<ReplayWire>("/api/replay", { method: "POST", body: JSON.stringify(doc) })
  );
}

// Load a finished game for replay (409 while it is still running).
export async function loadReplayFromGame(gameId: string): Promise<ReplayState> {
  return adapt(await api<ReplayWire>(`/api/games/${gameId}/replay`, { method: "POST" }));
}

export async function fetchReplayState(move: number): Promise<ReplayState> {
  return adapt(await api<ReplayWire>(`/api/replay/state?move=${move}`));
}

// The replay currently loaded on the server (one server-wide), or null. Unlike
// fetchReplayState this never 404s, so it's the safe page-load probe.
export async function currentReplay(): Promise<ReplayState | null> {
  const wire = await api<ReplayWire | null>("/api/replay");
  return wire ? adapt(wire) : null;
}
