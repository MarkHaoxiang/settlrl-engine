// The seat tokens this browser holds, per game id. A token is the bearer
// proof of one claimed seat (multiplayer identity); a hotseat client holds
// one per local seat. Persisted so reloads and invite links keep your seats.

export type SeatTokens = Record<number, string>;

const KEY = "settlrl-seats";
// The one place this browser is currently in — a pre-game lobby or a live game
// (a guest has no account, so this is the only way to hold them to one at a
// time). Set on create/join, cleared on leave or when the game ends.
const CURRENT = "settlrl-current";

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

export type CurrentPlace = { id: string; kind: "lobby" | "game" };

export function currentPlace(): CurrentPlace | null {
  try {
    const raw = localStorage.getItem(CURRENT);
    return raw ? (JSON.parse(raw) as CurrentPlace) : null;
  } catch {
    return null;
  }
}

export function setCurrentPlace(id: string, kind: "lobby" | "game"): void {
  localStorage.setItem(CURRENT, JSON.stringify({ id, kind }));
}

// Forget the current place. Pass an id to clear only if it still matches (so a
// stale finished game doesn't wipe a newer place the user has since joined).
export function clearCurrentPlace(id?: string): void {
  if (id === undefined || currentPlace()?.id === id) localStorage.removeItem(CURRENT);
}
