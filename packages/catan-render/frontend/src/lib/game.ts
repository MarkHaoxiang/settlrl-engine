// Live-game API client (/api/game*): snapshot types + the three calls.
//
// A snapshot is the server's GameModel — board + turn status + the human's
// legal moves, each decoded into a GameAction carrying its flat engine index
// (post it back to apply the move) and whatever geometry / resource choice it
// targets.

import { api } from "./api";
import { adaptBoard, type Board, type BoardWire } from "./boardData";
import type { Cube, CubeEdge, Hex } from "./hex";

export interface GameAction {
  flat: number;
  type: string;
  label: string;
  // Placement target (at most one group is set, depending on `type`).
  vertex: Cube | null;
  edge: CubeEdge | null;
  tile: Hex | null;
  victim: number | null;
  // Resource choices: monopoly / year-of-plenty / maritime trade.
  resource: string | null;
  resources: string[] | null;
  give: string | null;
  receive: string | null;
}

export interface GameStatus {
  phase: string;
  current_player: number;
  acting_player: number;
  dice_roll: number;
  has_rolled: boolean;
  your_turn: boolean;
  terminal: boolean;
  winner: number | null;
  // What controls each seat: "human" or a bot kind (see fetchBots).
  seats: string[];
}

// A bot move just played by the server (set on bot-step snapshots).
export interface BotMove {
  player: number;
  action: GameAction;
}

// One line of the server-side game log: a played move, a chat message, or the
// win. `player` is the seat it belongs to (null: a spectator's chat message);
// moves carry an action_type the client maps to an icon.
export interface LogEntry {
  id: number;
  kind: "move" | "chat" | "win";
  player: number | null;
  action_type: string | null;
  text: string;
}

export interface GameSnapshot {
  board: Board;
  status: GameStatus;
  actions: GameAction[];
  bot_move: BotMove | null;
  log: LogEntry[];
}

interface GameWire {
  board: BoardWire;
  status: GameStatus;
  actions: GameAction[];
  bot_move: BotMove | null;
  log: LogEntry[];
}

const adaptGame = (wire: GameWire): GameSnapshot => ({
  ...wire,
  board: adaptBoard(wire.board),
});

export async function fetchGame(): Promise<GameSnapshot> {
  return adaptGame(await api<GameWire>("/api/game"));
}

export async function postAction(flat: number): Promise<GameSnapshot> {
  return adaptGame(
    await api<GameWire>("/api/game/action", {
      method: "POST",
      body: JSON.stringify({ flat }),
    })
  );
}

// Step one due bot move; the snapshot's bot_move says what was played (null
// when no bot move was due).
export async function postBotStep(): Promise<GameSnapshot> {
  return adaptGame(await api<GameWire>("/api/game/bot", { method: "POST" }));
}

// Append a chat message to the game log (player: the seat it belongs to;
// null for a spectator).
export async function postChat(text: string, player: number | null): Promise<GameSnapshot> {
  return adaptGame(
    await api<GameWire>("/api/game/chat", {
      method: "POST",
      body: JSON.stringify({ text, player }),
    })
  );
}

// Seats in the new game; the renderer offers 2 and 4 for now.
export type PlayerCount = 2 | 4;
export type NumberPlacement = "random" | "spiral";
// What controls a seat: "human" (hotseat) or a bot kind from fetchBots.
export type SeatKind = string;
export const HUMAN: SeatKind = "human";

export interface NewGameConfig {
  seed: number;
  nPlayers: PlayerCount;
  numberPlacement: NumberPlacement;
  // One entry per seat; no seat has to be human (an all-bot game spectates).
  seats: SeatKind[];
}

export async function postReset(config: NewGameConfig): Promise<GameSnapshot> {
  return adaptGame(
    await api<GameWire>("/api/game/reset", {
      method: "POST",
      body: JSON.stringify({
        seed: config.seed,
        n_players: config.nPlayers,
        number_placement: config.numberPlacement,
        seats: config.seats,
      }),
    })
  );
}

// The bot kinds the server offers for seats, each with the player counts it
// supports (the two-player search agents are 2-only).
export async function fetchBots(): Promise<Record<string, number[]>> {
  return api<Record<string, number[]>>("/api/bots");
}
