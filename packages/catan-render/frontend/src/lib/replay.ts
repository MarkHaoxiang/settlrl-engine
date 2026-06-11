// Replay API client (/api/replay*): load a game record on the server and
// fetch the board snapshot + move log at any point of the game.

import { api } from "./api";
import { adaptBoard, type Board, type BoardWire } from "./boardData";
import type { LogEntry } from "./game";

export interface ReplayState {
  // Snapshot after `move` of `n_moves` moves (0 = the opening board).
  move: number;
  n_moves: number;
  board: Board;
  // Moves played up to this point; the win line appears only at the end.
  log: LogEntry[];
  winner: number | null;
  seats: string[] | null;
}

interface ReplayWire extends Omit<ReplayState, "board"> {
  board: BoardWire;
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
