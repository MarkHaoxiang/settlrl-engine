import { useEffect, useState } from "react";
import {
  listBotProviders,
  registerBotProvider,
  removeBotProvider,
  type BotProvider,
} from "../lib/admin";
import type { AuthUser } from "../lib/auth";
import { buttonStyle, panelStyle, smallButtonStyle } from "../lib/ui";

// Admin-only panel for managing the remote bot services whose kinds become the
// seatable bots. Hidden unless the signed-in user is a superuser. Registrations
// live in the server's memory, so they need re-adding after a server restart.
export default function BotProviders({ user }: { user: AuthUser | null }) {
  const [providers, setProviders] = useState<BotProvider[]>([]);
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const admin = !!user?.is_superuser;
  useEffect(() => {
    if (admin) void listBotProviders().then(setProviders).catch(() => setProviders([]));
    else setProviders([]);
  }, [admin]);

  if (!admin) return null;

  const add = async () => {
    setBusy(true);
    setError(null);
    try {
      await registerBotProvider(name.trim(), baseUrl.trim());
      setProviders(await listBotProviders());
      setName("");
      setBaseUrl("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (n: string) => {
    setError(null);
    try {
      await removeBotProvider(n);
      setProviders(await listBotProviders());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const inputStyle: React.CSSProperties = { ...panelStyle, padding: "8px 10px", fontSize: 13 };
  const canAdd = name.trim() !== "" && baseUrl.trim() !== "" && !busy;
  return (
    <div style={{ ...panelStyle, padding: "16px 20px", borderRadius: 12, minWidth: 300 }}>
      <div
        style={{
          fontSize: 12,
          opacity: 0.6,
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 10,
        }}
      >
        Bot services
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12 }}>
        {providers.length === 0 && (
          <span style={{ fontSize: 13, opacity: 0.6 }}>
            None registered — no bots are seatable yet.
          </span>
        )}
        {providers.map((p) => (
          <div
            key={p.name}
            style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}
          >
            <span style={{ flex: 1 }}>
              <b>{p.name}</b> <span style={{ opacity: 0.6 }}>{p.base_url}</span>
              <br />
              <span style={{ opacity: 0.6 }}>{p.kinds.join(", ")}</span>
            </span>
            <button style={smallButtonStyle} onClick={() => void remove(p.name)}>
              Remove
            </button>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <input
          style={inputStyle}
          placeholder="name (e.g. local)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          style={inputStyle}
          placeholder="base URL (e.g. http://localhost:8100)"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && canAdd && void add()}
        />
        {error && <span style={{ color: "var(--error)", fontSize: 12 }}>{error}</span>}
        <button style={buttonStyle} disabled={!canAdd} onClick={() => void add()}>
          Register service
        </button>
      </div>
    </div>
  );
}
