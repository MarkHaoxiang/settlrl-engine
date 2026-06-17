// Live-game API client (/api/games*): snapshot types + the calls.
//
// A snapshot is the server's GameModel as one requester sees it — board +
// turn status + their legal moves (each a GameAction carrying its flat engine
// index) — scoped by the seat tokens sent with every request: your_turn,
// actions, unredacted hands, and belief all follow the seats you can prove.

import { api, sse } from "./api";
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

// Seats in the new game; the renderer offers 2 and 4 for now.
export type PlayerCount = 2 | 4;
export type NumberPlacement = "random" | "spiral";
// What controls a seat: "human" (hotseat) or a bot kind from fetchBots.
export type SeatKind = string;
export const HUMAN: SeatKind = "human";

// A configurable scalar build parameter of a bot kind, and its value.
export type BotParamValue = number | boolean;

// A seat assignment: its controller plus any bot knob overrides
// (params only ever holds values the user changed from the defaults).
export interface SeatConfig {
  kind: SeatKind;
  params?: Record<string, BotParamValue>;
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
      // Plain kind strings unless a seat carries knob overrides.
      seats: config.seats.map((s) =>
        s.params && Object.keys(s.params).length > 0 ? { kind: s.kind, params: s.params } : s.kind
      ),
      claim: config.claim,
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

// One knob of a bot kind, as described by GET /api/bots.
export interface BotParamSpec {
  type: "int" | "float" | "bool";
  default: BotParamValue;
}

// A bot kind's catalog entry: the player counts it supports, a short
// description, and its tunable knobs.
export interface BotSpec {
  counts: number[];
  description: string;
  params: Record<string, BotParamSpec>;
}

export async function fetchBots(): Promise<Record<string, BotSpec>> {
  return api<Record<string, BotSpec>>("/api/bots");
}
