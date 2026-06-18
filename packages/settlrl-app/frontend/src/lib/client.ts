// One typed REST client for the whole API, generated from the committed schema
// (openapi-fetch): paths, params, and response types are checked against the
// server, so calls can't drift from the contract. The account bearer token is
// injected on every request, so callers never assemble auth headers by hand.

import createClient from "openapi-fetch";

import { API_BASE } from "./api";
import { authToken } from "./auth";
import type { paths } from "./api-schema";

export const client = createClient<paths>({ baseUrl: API_BASE });

client.use({
  onRequest({ request }) {
    const token = authToken();
    if (token) request.headers.set("Authorization", `Bearer ${token}`);
    return request;
  },
});

// Throw on a non-2xx so React Query sees an error; return the typed body.
export async function unwrap<T>(
  req: Promise<{ data?: T; error?: unknown }>
): Promise<T> {
  const { data, error } = await req;
  if (error !== undefined || data === undefined)
    throw error ?? new Error("request failed");
  return data;
}
