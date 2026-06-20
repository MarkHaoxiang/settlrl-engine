import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Button from "../components/Button";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import {
  getAdminStatus,
  registerBotProvider,
  removeBotProvider,
  type AdminStatus,
} from "../lib/admin";
import { currentUser, type AuthUser } from "../lib/auth";
import ui from "../styles/ui.module.css";
import s from "./AdminView.module.css";

const POLL_MS = 4000;

const fmtUptime = (secs: number): string => {
  const s = Math.max(0, Math.round(secs));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
};

const ago = (epoch: number): string => {
  const secs = Math.max(0, Math.round(Date.now() / 1000 - epoch));
  if (secs < 60) return `${secs}s`;
  const mins = Math.round(secs / 60);
  return mins < 60 ? `${mins}m` : `${Math.round(mins / 60)}h`;
};

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className={s.stat}>
      <span className={s.statValue}>{value}</span>
      <span className={s.statLabel}>{label}</span>
    </div>
  );
}

export default function AdminView() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The base URL being typed into the register-a-bot-service form, plus its
  // own error (separate from the status-poll error).
  const [baseUrl, setBaseUrl] = useState("");
  const [botError, setBotError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  // Gate the page on the superuser flag (the API enforces it too); bounce others.
  useEffect(() => {
    void currentUser().then((u) => {
      setUser(u);
      if (!u?.is_superuser) navigate("/", { replace: true });
    });
  }, [navigate]);

  const refresh = useCallback(
    () =>
      getAdminStatus().then(
        (st) => (setStatus(st), setError(null)),
        (e: unknown) => setError(String(e))
      ),
    []
  );

  useEffect(() => {
    let live = true;
    const tick = () => (live ? refresh() : Promise.resolve());
    void tick();
    const t = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      live = false;
      window.clearInterval(t);
    };
  }, [refresh]);

  // Registering/removing a bot service mutates the catalog, then re-pulls the
  // status so the panel reflects it immediately (rather than waiting a poll).
  const addProvider = async () => {
    setBusy(true);
    setBotError(null);
    try {
      await registerBotProvider(baseUrl.trim());
      setBaseUrl("");
      await refresh();
    } catch (e) {
      setBotError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const dropProvider = async (name: string) => {
    setBotError(null);
    try {
      await removeBotProvider(name);
      await refresh();
    } catch (e) {
      setBotError(e instanceof Error ? e.message : String(e));
    }
  };
  const canAdd = baseUrl.trim() !== "" && !busy;

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/" className={ui.backLink}>
        ‹ Menu
      </Link>

      <h1 className={s.title}>Admin</h1>

      {error && <span className={s.error}>{error}</span>}
      {!status ? (
        <span className={s.muted}>Loading…</span>
      ) : (
        <>
          <Panel className={s.box}>
            <span className={ui.sectionLabel}>Server</span>
            <div className={s.stats}>
              <Stat label="uptime" value={fmtUptime(status.uptime_seconds)} />
              <Stat label="active games" value={status.games_active} />
              <Stat label="games held" value={`${status.games_total} / ${status.games_capacity}`} />
              <Stat label="bot kinds" value={status.bot_kinds.length} />
            </div>
          </Panel>

          <Panel className={s.box}>
            <span className={ui.sectionLabel}>Bot services</span>
            {status.bot_providers.length === 0 ? (
              <span className={s.muted}>None registered — no bots are seatable yet.</span>
            ) : (
              status.bot_providers.map((p) => (
                <div key={String(p.name)} className={s.providerRow}>
                  <span className={s.providerInfo}>
                    <span className={s.providerName}>{String(p.name)}</span>
                    <span className={s.muted}>{String(p.base_url)}</span>
                  </span>
                  <Button variant="small" onClick={() => void dropProvider(String(p.name))}>
                    Remove
                  </Button>
                </div>
              ))
            )}
            {status.bot_kinds.length > 0 && (
              <div className={s.kinds}>
                {status.bot_kinds.map((k) => (
                  <span key={k} className={s.kindTag}>
                    {k}
                  </span>
                ))}
              </div>
            )}
            <div className={s.form}>
              <input
                className={s.input}
                placeholder="base URL (e.g. http://localhost:8100)"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && canAdd && void addProvider()}
              />
              <Button disabled={!canAdd} onClick={() => void addProvider()}>
                Register service
              </Button>
            </div>
            {botError && <span className={s.error}>{botError}</span>}
          </Panel>

          <Panel className={s.box}>
            <span className={ui.sectionLabel}>Games ({status.games.length})</span>
            {status.games.length === 0 ? (
              <span className={s.muted}>No live games.</span>
            ) : (
              <table className={s.table}>
                <thead>
                  <tr className={s.headRow}>
                    <th className={s.cell}>id</th>
                    <th className={s.cellRight}>players</th>
                    <th className={s.cell}>phase</th>
                    <th className={s.cellRight}>moves</th>
                    <th className={s.cellRight}>open</th>
                    <th className={s.cell}>flags</th>
                    <th className={s.cellRight}>age</th>
                  </tr>
                </thead>
                <tbody>
                  {status.games.map((g) => (
                    <tr key={g.id} className={s.row}>
                      <td className={s.mono}>{g.id.slice(0, 8)}</td>
                      <td className={s.cellRight}>{g.n_players}</td>
                      <td className={s.cell}>{g.terminal ? "over" : g.phase}</td>
                      <td className={s.cellRight}>{g.moves}</td>
                      <td className={s.cellRight}>{g.open_seats}</td>
                      <td className={s.cell}>
                        {[g.listed && "listed", g.searchable && "QM"].filter(Boolean).join(" ") || "—"}
                      </td>
                      <td className={s.cellRight}>{ago(g.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}
