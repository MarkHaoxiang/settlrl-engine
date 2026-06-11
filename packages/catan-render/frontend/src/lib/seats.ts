// The seat tokens this browser holds, per game id. A token is the bearer
// proof of one claimed seat (multiplayer identity); a hotseat client holds
// one per local seat. Persisted so reloads and invite links keep your seats.

export type SeatTokens = Record<number, string>;

const KEY = "catan-seats";
const LAST = "catan-last-game";

function readAll(): Record<string, SeatTokens> {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}") as Record<string, SeatTokens>;
  } catch {
    return {};
  }
}

export function tokensFor(gameId: string): SeatTokens {
  return readAll()[gameId] ?? {};
}

export function saveTokens(gameId: string, tokens: SeatTokens): void {
  const all = readAll();
  all[gameId] = { ...all[gameId], ...tokens };
  localStorage.setItem(KEY, JSON.stringify(all));
}

// The most recently played game (the Replay view's "use current game").
export function rememberGame(gameId: string): void {
  localStorage.setItem(LAST, gameId);
}

export function lastGameId(): string | null {
  return localStorage.getItem(LAST);
}
