// Live-game API client (/api/games*): snapshot types + the calls.
//
// A snapshot is the server's GameModel as one requester sees it — board +
// turn status + their legal moves (each a GameAction carrying its flat engine
// index) — scoped by the seat tokens sent with every request: your_turn,
// actions, unredacted hands, and belief all follow the seats you can prove.

import { API_BASE, ApiError, api, sse } from "./api";
import type { components } from "./api-schema";
import { authHeader } from "./auth";
import { adaptBoard, type Board } from "./boardData";
import type { SeatTokens } from "./seats";

type Schemas = components["schemas"];

// Every game request carries the seat tokens this device holds *and* the
// account bearer token (if signed in), so the server recognises seats owned by
// token or by account. authHeader() is empty when logged out, so the anonymous
// path is unchanged.
const seatHeaders = (tokens: SeatTokens): Record<string, string> => {
  const values = Object.values(tokens);
  return {
    ...(values.length ? { "X-Seat-Tokens": values.join(",") } : {}),
    ...authHeader(),
  };
};

// One legal move: the flat engine index plus whatever geometry / resource
// choice it targets (generated from the server's ActionModel).
export type GameAction = Schemas["ActionModel"];

export type TradeOffer = Schemas["TradeOfferModel"];
export type GameStatus = Schemas["GameStatusModel"];
export type BotMove = Schemas["BotMoveModel"];
export type LogEntry = Schemas["LogEntryModel"];
export type PlayerBelief = Schemas["PlayerBeliefModel"];
export type Belief = Schemas["BeliefModel"];

type GameWire = Schemas["GameModel"];

// The wire snapshot with its board adapted to frontend coordinates.
export interface GameSnapshot extends Omit<GameWire, "board"> {
  board: Board;
}

const adaptGame = (wire: GameWire): GameSnapshot => ({
  ...wire,
  board: adaptBoard(wire.board),
});

// The board a new game would open on (no game created), for the map picker.
export async function fetchPreview(
  seed: number,
  nPlayers: PlayerCount,
  numberPlacement: NumberPlacement
): Promise<Board> {
  const q = new URLSearchParams({
    seed: String(seed),
    n_players: String(nPlayers),
    number_placement: numberPlacement,
  });
  return adaptBoard(await api<Schemas["BoardModel"]>(`/api/preview?${q}`));
}

export async function fetchGame(gameId: string, tokens: SeatTokens): Promise<GameSnapshot> {
  return adaptGame(await api<GameWire>(`/api/games/${gameId}`, { headers: seatHeaders(tokens) }));
}

export async function postAction(
  gameId: string,
  tokens: SeatTokens,
  flat: number
): Promise<GameSnapshot> {
  return adaptGame(
    await api<GameWire>(`/api/games/${gameId}/action`, {
      method: "POST",
      body: JSON.stringify({ flat }),
      headers: seatHeaders(tokens),
    })
  );
}

// Subscribe to the game's pushed snapshots (SSE): the current one arrives
// immediately, then a new one on every state change — moves, server-paced
// bot plays, chat, joins. Ends only on abort or a dropped connection.
export async function* streamGame(
  gameId: string,
  tokens: SeatTokens,
  signal: AbortSignal
): AsyncGenerator<GameSnapshot> {
  for await (const data of sse(`/api/games/${gameId}/events`, seatHeaders(tokens), signal)) {
    yield adaptGame(JSON.parse(data) as GameWire);
  }
}

// Append a chat message to the game log (player: an owned seat it belongs
// to; null for a spectator).
export async function postChat(
  gameId: string,
  tokens: SeatTokens,
  text: string,
  player: number | null
): Promise<GameSnapshot> {
  return adaptGame(
    await api<GameWire>(`/api/games/${gameId}/chat`, {
      method: "POST",
      body: JSON.stringify({ text, player }),
      headers: seatHeaders(tokens),
    })
  );
}

// Seats in the new game; the app offers 2 and 4 for now.
export type PlayerCount = 2 | 4;
export type NumberPlacement = "random" | "spiral";
// What controls a seat: "human" (hotseat) or a bot kind from fetchBots.
export type SeatKind = string;
export const HUMAN: SeatKind = "human";

// A seat assignment: just its controller (one bot service = one configured bot).
export interface SeatConfig {
  kind: SeatKind;
}

// Which human seats the creator claims: every one (hotseat, sharing this
// screen) or just the first (online — the others join via the invite link).
export type ClaimMode = "all" | "first";

export interface NewGameConfig {
  seed: number;
  nPlayers: PlayerCount;
  numberPlacement: NumberPlacement;
  // One entry per seat; no seat has to be human (an all-bot game spectates).
  seats: SeatConfig[];
  claim: ClaimMode;
  // List the game in the public lobby so anyone can join its open seats.
  listed: boolean;
}

// A freshly created game: its id plus the creator's seat tokens (every human
// seat is claimed for the hotseat default).
export interface CreatedGame {
  id: string;
  seats: string[];
  tokens: SeatTokens;
}

// The server is at its concurrency cap: the caller's place in line. Re-call
// createGame with the ticket to keep polling until a CreatedGame comes back.
export interface QueuedGame {
  queued: true;
  ticket: string;
  position: number;
  total: number;
}

export async function createGame(
  config: NewGameConfig,
  ticket?: string
): Promise<CreatedGame | QueuedGame> {
  return api<CreatedGame | QueuedGame>("/api/games", {
    method: "POST",
    body: JSON.stringify({
      seed: config.seed,
      n_players: config.nPlayers,
      number_placement: config.numberPlacement,
      seats: config.seats.map((s) => s.kind),
      claim: config.claim,
      listed: config.listed,
      ticket,
    }),
    // Sign-in (if any) ties the creator's claimed seats to their account.
    headers: authHeader(),
  });
}

// Claim a human seat in an existing game (a specific one, or the first free).
export async function joinGame(
  gameId: string,
  seat?: number
): Promise<{ id: string; seat: number; token: string }> {
  return api(`/api/games/${gameId}/join`, {
    method: "POST",
    body: JSON.stringify(seat == null ? {} : { seat }),
    // Sign-in (if any) ties the claimed seat to the account.
    headers: authHeader(),
  });
}

// Retarget a still-open seat (open <-> a bot kind). Owner-only, pre-start; the
// returned snapshot reflects the new seat kinds. Used by the waiting room to
// bot-fill seats so an under-filled game can start.
export async function setSeat(
  gameId: string,
  tokens: SeatTokens,
  seat: number,
  kind: SeatKind
): Promise<GameSnapshot> {
  return adaptGame(
    await api<GameWire>(`/api/games/${gameId}/seats`, {
      method: "POST",
      body: JSON.stringify({ seat, kind }),
      headers: seatHeaders(tokens),
    })
  );
}

// Still finding an Elo match: re-POST with the ticket to keep polling.
export interface MatchQueued {
  queued: true;
  ticket: string;
  waiting: number;
}

// Matched: a claimed seat in a freshly created game.
export interface MatchFound {
  id: string;
  seat: number;
  token: string;
}

// One Elo Quick Match poll. Pass the prior ticket to keep your place; the result
// is either a queued position (re-poll) or the seat you were matched into.
export async function matchmake(
  nPlayers: PlayerCount,
  ticket?: string
): Promise<MatchQueued | MatchFound> {
  return api<MatchQueued | MatchFound>("/api/matchmake", {
    method: "POST",
    body: JSON.stringify({ n_players: nPlayers, ticket }),
    headers: authHeader(),
  });
}

// A bot's catalog entry (one per registered bot service): a display title, a
// short description, and the player counts it supports.
export interface BotSpec {
  title?: string;
  description: string;
  counts: number[];
}

export async function fetchBots(): Promise<Record<string, BotSpec>> {
  return api<Record<string, BotSpec>>("/api/bots");
}

// Download a finished game's replay (the GameRecord JSON) as a file the player
// can keep and re-load in the Replay view.
export async function downloadRecord(gameId: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/games/${gameId}/record`, {
    headers: authHeader(),
  });
  if (!resp.ok) throw new ApiError(resp.status, resp.statusText);
  const url = URL.createObjectURL(new Blob([await resp.text()], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `settlrl-replay-${gameId}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
