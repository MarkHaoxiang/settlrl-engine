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

// Resume links carry seat tokens so you can restore your seats on another
// device (or after clearing storage). Tokens are url-safe base64 — no ":" or
// "," — so "seat:token" pairs joined by "," round-trip unambiguously.
export function encodeTokens(tokens: SeatTokens): string {
  return Object.entries(tokens)
    .map(([seat, token]) => `${seat}:${token}`)
    .join(",");
}

export function parseTokens(value: string): SeatTokens {
  const out: SeatTokens = {};
  for (const pair of value.split(",")) {
    const i = pair.indexOf(":");
    if (i <= 0) continue;
    const seat = Number(pair.slice(0, i));
    const token = pair.slice(i + 1);
    if (Number.isInteger(seat) && token) out[seat] = token;
  }
  return out;
}

// A link that restores the held seats when opened (consumed by the Play view).
export function resumeLink(gameId: string, tokens: SeatTokens): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  const q = encodeURIComponent(encodeTokens(tokens));
  return `${window.location.origin}${base}/play/${gameId}?tokens=${q}`;
}

// The most recently played game (the Replay view's "use current game").
export function rememberGame(gameId: string): void {
  localStorage.setItem(LAST, gameId);
}

export function lastGameId(): string | null {
  return localStorage.getItem(LAST);
}
