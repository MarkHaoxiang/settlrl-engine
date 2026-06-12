// Minimal JSON fetch wrapper for the renderer's API, plus the SSE reader.
// All paths are prefixed with the build-time base (vite `--base`), so the
// app works at / and behind a stripped proxy prefix (e.g. /catan) alike.

export const API_BASE = import.meta.env.BASE_URL.replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    detail: string
  ) {
    super(detail);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(API_BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!resp.ok) {
    const detail = await resp
      .json()
      .then((body) => String(body.detail ?? resp.statusText))
      .catch(() => resp.statusText);
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

// Subscribe to a server-sent-event stream (fetch-based: EventSource can't
// send headers), yielding each event's data payload. Ends when the server
// closes the stream or `signal` aborts; throws ApiError on a bad status.
export async function* sse(
  path: string,
  headers: Record<string, string>,
  signal: AbortSignal
): AsyncGenerator<string> {
  const resp = await fetch(API_BASE + path, { headers, signal });
  if (!resp.ok || !resp.body) throw new ApiError(resp.status, resp.statusText);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let end;
    while ((end = buf.indexOf("\n\n")) >= 0) {
      const event = buf.slice(0, end);
      buf = buf.slice(end + 2);
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (data) yield data;
    }
  }
}
