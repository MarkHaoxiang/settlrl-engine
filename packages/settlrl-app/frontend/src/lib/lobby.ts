// Lobby API client (/api/lobbies*): the pre-game staging table that becomes a
// game on start. A lobby holds config + claimed seats and a board *preview*; it
// has no GameSession until the host starts it (which returns the new game id).

import { api, sse } from "./api";
import { authHeader } from "./auth";
import type { components } from "./api-schema";
import { adaptBoard, type Board } from "./boardData";
import type { SeatTokens } from "./seats";

type Schemas = components["schemas"];

// Seat tokens *and* the account bearer (if any), so the server recognises seats
// owned by token or by account.
const seatHeaders = (tokens: SeatTokens): Record<string, string> => {
  const values = Object.values(tokens);
  return {
    ...(values.length ? { "X-Seat-Tokens": values.join(",") } : {}),
    ...authHeader(),
  };
};

type LobbyWire = Schemas["_LobbyModel"];
export type LobbyListing = Schemas["_LobbyListModel"];
export type LobbyMode = "hotseat" | "online";

// The wire snapshot with its preview board adapted to frontend coordinates.
export interface LobbySnapshot extends Omit<LobbyWire, "board"> {
  board: Board;
}
const adapt = (wire: LobbyWire): LobbySnapshot => ({ ...wire, board: adaptBoard(wire.board) });

export interface CreatedLobby {
  id: string;
  tokens: Record<number, string>;
}

export async function createLobby(req: {
  mode: LobbyMode;
  nPlayers?: number;
  listed?: boolean;
}): Promise<CreatedLobby> {
  return api("/api/lobbies", {
    method: "POST",
    body: JSON.stringify({ mode: req.mode, n_players: req.nPlayers ?? 4, listed: req.listed }),
    headers: authHeader(),
  });
}

export async function listLobbies(): Promise<LobbyListing[]> {
  return api("/api/lobbies");
}

export async function getLobby(id: string, tokens: SeatTokens): Promise<LobbySnapshot> {
  return adapt(await api<LobbyWire>(`/api/lobbies/${id}`, { headers: seatHeaders(tokens) }));
}

export async function* streamLobby(
  id: string,
  tokens: SeatTokens,
  signal: AbortSignal
): AsyncGenerator<LobbySnapshot> {
  for await (const data of sse(`/api/lobbies/${id}/events`, seatHeaders(tokens), signal)) {
    yield adapt(JSON.parse(data) as LobbyWire);
  }
}

export async function configureLobby(
  id: string,
  tokens: SeatTokens,
  config: {
    seed?: number;
    nPlayers?: number;
    numberPlacement?: string;
    victoryPointsToWin?: number;
    listed?: boolean;
    searchable?: boolean;
  }
): Promise<LobbySnapshot> {
  return adapt(
    await api<LobbyWire>(`/api/lobbies/${id}/configure`, {
      method: "POST",
      body: JSON.stringify({
        seed: config.seed,
        n_players: config.nPlayers,
        number_placement: config.numberPlacement,
        victory_points_to_win: config.victoryPointsToWin,
        listed: config.listed,
        searchable: config.searchable,
      }),
      headers: seatHeaders(tokens),
    })
  );
}

export async function setLobbySeat(
  id: string,
  tokens: SeatTokens,
  seat: number,
  kind: string
): Promise<LobbySnapshot> {
  return adapt(
    await api<LobbyWire>(`/api/lobbies/${id}/seats`, {
      method: "POST",
      body: JSON.stringify({ seat, kind }),
      headers: seatHeaders(tokens),
    })
  );
}

export async function joinLobby(id: string, seat?: number): Promise<CreatedLobby> {
  return api(`/api/lobbies/${id}/join`, {
    method: "POST",
    body: JSON.stringify(seat == null ? {} : { seat }),
    headers: authHeader(),
  });
}

export async function leaveLobby(id: string, tokens: SeatTokens): Promise<void> {
  await api(`/api/lobbies/${id}/leave`, { method: "POST", headers: seatHeaders(tokens) });
}

export async function chatLobby(
  id: string,
  tokens: SeatTokens,
  text: string,
  player: number | null
): Promise<LobbySnapshot> {
  return adapt(
    await api<LobbyWire>(`/api/lobbies/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ text, player }),
      headers: seatHeaders(tokens),
    })
  );
}

export type StartResult = { game_id: string } | { queued: true; position: number; total: number };

export async function startLobby(
  id: string,
  tokens: SeatTokens,
  ticket?: string
): Promise<StartResult> {
  return api(`/api/lobbies/${id}/start`, {
    method: "POST",
    body: JSON.stringify(ticket ? { ticket } : {}),
    headers: seatHeaders(tokens),
  });
}
