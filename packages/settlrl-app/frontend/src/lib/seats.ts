// The seat tokens this browser holds, per game id. A token is the bearer
// proof of one claimed seat (multiplayer identity); a hotseat client holds
// one per local seat. Persisted so reloads and invite links keep your seats.

export type SeatTokens = Record<number, string>;

const KEY = "settlrl-seats";
// The one live game this browser is currently in (a guest has no account, so
// this is the only way to hold them to one game at a time). Set on create/join,
// cleared on leave or when the game ends.
const CURRENT = "settlrl-current-game";

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

export function currentGameId(): string | null {
  return localStorage.getItem(CURRENT);
}

export function setCurrentGame(gameId: string): void {
  localStorage.setItem(CURRENT, gameId);
}

// Forget the current game. Pass an id to clear only if it still matches (so a
// stale finished game doesn't wipe a newer one the user has since joined).
export function clearCurrentGame(gameId?: string): void {
  if (gameId === undefined || localStorage.getItem(CURRENT) === gameId) {
    localStorage.removeItem(CURRENT);
  }
}
