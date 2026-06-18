// Account / auth client (/api/auth*). Optional: gameplay stays anonymous (seat
// tokens), so being logged out changes nothing about playing. A login mints a
// bearer token kept in localStorage; admin-only screens send it.

import { API_BASE, ApiError, api } from "./api";

export interface AuthUser {
  id: string;
  email: string;
  is_superuser: boolean;
}

const TOKEN_KEY = "settlrl-auth-token";

export const authToken = (): string | null => localStorage.getItem(TOKEN_KEY);
const setToken = (token: string | null): void => {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
};

// The Authorization header for an authenticated request (empty when logged out).
export const authHeader = (): Record<string, string> => {
  const token = authToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

export async function register(email: string, password: string): Promise<AuthUser> {
  return api<AuthUser>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function login(email: string, password: string): Promise<AuthUser> {
  // The OAuth2 password flow takes a form-encoded body, not JSON.
  const resp = await fetch(API_BASE + "/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: email, password }),
  });
  if (!resp.ok) {
    const detail = await resp
      .json()
      .then((b) => String(b.detail ?? resp.statusText))
      .catch(() => resp.statusText);
    throw new ApiError(resp.status, detail);
  }
  // fastapi-users login returns only the bearer token; fetch the user with it.
  const data = (await resp.json()) as { access_token: string };
  setToken(data.access_token);
  const user = await currentUser();
  if (!user) throw new ApiError(500, "logged in but could not load the account");
  return user;
}

export async function logout(): Promise<void> {
  const token = authToken();
  if (token) {
    await fetch(API_BASE + "/api/auth/logout", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    }).catch(() => undefined);
  }
  setToken(null);
}

// The signed-in user's live games and finished history are React Query hooks
// over the typed client (lib/queries: useMyGames / useHistory).

// The signed-in user for the stored token, or null (clearing a dead token).
export async function currentUser(): Promise<AuthUser | null> {
  const token = authToken();
  if (!token) return null;
  const resp = await fetch(API_BASE + "/api/users/me", {
    headers: { Authorization: `Bearer ${token}` },
  }).catch(() => null);
  if (!resp || !resp.ok) {
    if (resp && resp.status === 401) setToken(null);
    return null;
  }
  return (await resp.json()) as AuthUser;
}
