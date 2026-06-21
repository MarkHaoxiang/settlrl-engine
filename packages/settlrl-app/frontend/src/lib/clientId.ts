// A stable anonymous id for this browser, persisted across reloads. Sent as the
// X-Client-Id header so the server holds a guest (no account) to one game at a
// time and never matches them against themselves in Quick Match. Not a secret
// and not an ownership proof — only an identity hint.

const KEY = "settlrl-client-id";

export function clientId(): string {
  let id = localStorage.getItem(KEY);
  if (!id) {
    // crypto.randomUUID needs a secure context (https / localhost); fall back so
    // a plain-http LAN host still gets a usable id.
    id = crypto.randomUUID?.() ?? `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem(KEY, id);
  }
  return id;
}
