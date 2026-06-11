// Minimal JSON fetch wrapper for the renderer's API.

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    detail: string
  ) {
    super(detail);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
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
