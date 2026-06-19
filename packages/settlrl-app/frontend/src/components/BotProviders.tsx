import { useEffect, useState } from "react";
import {
  listBotProviders,
  registerBotProvider,
  removeBotProvider,
  type BotProvider,
} from "../lib/admin";
import type { AuthUser } from "../lib/auth";
import Button from "./Button";
import s from "./BotProviders.module.css";

// Admin-only panel for managing the remote bot services that become the
// seatable bots (one bot per service). Hidden unless the signed-in user is a
// superuser. Registrations live in the server's memory, so they need re-adding
// after a server restart.
export default function BotProviders({ user }: { user: AuthUser | null }) {
  const [providers, setProviders] = useState<BotProvider[]>([]);
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
      await registerBotProvider(baseUrl.trim());
      setProviders(await listBotProviders());
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

  const canAdd = baseUrl.trim() !== "" && !busy;
  return (
    <div className={s.box}>
      <div className={s.label}>Bot services</div>
      <div className={s.list}>
        {providers.length === 0 && (
          <span className={s.empty}>None registered — no bots are seatable yet.</span>
        )}
        {providers.map((p) => (
          <div key={p.name} className={s.row}>
            <span className={s.info}>
              <b>{p.name}</b> <span className={s.url}>{p.base_url}</span>
            </span>
            <Button variant="small" onClick={() => void remove(p.name)}>
              Remove
            </Button>
          </div>
        ))}
      </div>
      <div className={s.form}>
        <input
          className={s.input}
          placeholder="base URL (e.g. http://localhost:8100)"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && canAdd && void add()}
        />
        {error && <span className={s.error}>{error}</span>}
        <Button disabled={!canAdd} onClick={() => void add()}>
          Register service
        </Button>
      </div>
    </div>
  );
}
