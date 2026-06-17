// Admin client (/api/admin/*): registering the remote bot services whose kinds
// form the seatable-bot catalog. Every call carries the bearer token; the
// server gates these on the superuser flag (403 otherwise).

import { API_BASE, ApiError, api } from "./api";
import { authHeader } from "./auth";

export interface BotProvider {
  name: string;
  base_url: string;
  kinds: string[];
}

export async function listBotProviders(): Promise<BotProvider[]> {
  return api<BotProvider[]>("/api/admin/bot-providers", { headers: authHeader() });
}

export async function registerBotProvider(name: string, baseUrl: string): Promise<BotProvider> {
  return api<BotProvider>("/api/admin/bot-providers", {
    method: "POST",
    headers: authHeader(),
    body: JSON.stringify({ name, base_url: baseUrl }),
  });
}

// DELETE returns 204 with no body, so it bypasses the JSON `api` helper.
export async function removeBotProvider(name: string): Promise<void> {
  const resp = await fetch(`${API_BASE}/api/admin/bot-providers/${encodeURIComponent(name)}`, {
    method: "DELETE",
    headers: authHeader(),
  });
  if (!resp.ok) throw new ApiError(resp.status, resp.statusText);
}
