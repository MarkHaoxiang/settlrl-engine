// Minimal JSON fetch wrapper for the app's API, plus the SSE reader.
// All paths are prefixed with the build-time base (vite `--base`), so the
// app works at / and behind a stripped proxy prefix (e.g. /settlrl) alike.

import { EventSourceParserStream } from "eventsource-parser/stream";
import { clientId } from "./clientId";

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
    headers: {
      "Content-Type": "application/json",
      "X-Client-Id": clientId(),
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    const detail = await resp
      .json()
      .then((body) => String(body.detail ?? resp.statusText))
      .catch(() => resp.statusText);
    throw new ApiError(resp.status, detail);
  }
  const text = await resp.text(); // tolerate an empty body (e.g. 204 No Content)
  return (text ? JSON.parse(text) : undefined) as T;
}

// Subscribe to a server-sent-event stream (fetch-based: EventSource can't
// send headers), yielding each event's data payload. Ends when the server
// closes the stream or `signal` aborts; throws ApiError on a bad status.
// Decoding and SSE framing (data lines, keepalive comments) are handled by
// eventsource-parser.
export async function* sse(
  path: string,
  headers: Record<string, string>,
  signal: AbortSignal
): AsyncGenerator<string> {
  const resp = await fetch(API_BASE + path, { headers, signal });
  if (!resp.ok || !resp.body) throw new ApiError(resp.status, resp.statusText);
  const events = resp.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream())
    .getReader();
  for (;;) {
    const { done, value } = await events.read();
    if (done) return;
    if (value.data) yield value.data;
  }
}
