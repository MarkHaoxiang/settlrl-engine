import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { getAdminStatus, type AdminStatus } from "../lib/admin";
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
  const navigate = useNavigate();

  // Gate the page on the superuser flag (the API enforces it too); bounce others.
  useEffect(() => {
    void currentUser().then((u) => {
      setUser(u);
      if (!u?.is_superuser) navigate("/", { replace: true });
    });
  }, [navigate]);

  useEffect(() => {
    let live = true;
    const tick = () =>
      getAdminStatus().then(
        (st) => live && (setStatus(st), setError(null)),
        (e: unknown) => live && setError(String(e))
      );
    void tick();
    const t = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      live = false;
      window.clearInterval(t);
    };
  }, []);

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
              <span className={s.muted}>None registered.</span>
            ) : (
              status.bot_providers.map((p) => (
                <div key={String(p.name)} className={s.providerRow}>
                  <span className={s.providerName}>{String(p.name)}</span>
                  <span className={s.muted}>{String(p.base_url)}</span>
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
