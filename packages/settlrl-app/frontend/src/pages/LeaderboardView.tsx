import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Button from "../components/Button";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import { useLeaderboard, type LeaderboardEntry } from "../lib/queries";
import ui from "../styles/ui.module.css";
import s from "./LeaderboardView.module.css";

const winRate = (e: LeaderboardEntry) =>
  e.games ? `${Math.round((100 * e.wins) / e.games)}%` : "—";

function Ladder({ rows }: { rows: LeaderboardEntry[] }) {
  if (rows.length === 0) return <span className={s.empty}>No rated games yet.</span>;
  return (
    <table className={s.table}>
      <thead>
        <tr className={s.headRow}>
          <th className={s.cell}>#</th>
          <th className={s.cell}>Player</th>
          <th className={s.cellRight}>Rating</th>
          <th className={s.cellRight}>Games</th>
          <th className={s.cellRight}>Win</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((e, i) => (
          <tr key={`${e.kind}:${e.name}`} className={s.row}>
            <td className={s.rank}>{i + 1}</td>
            <td className={s.cell}>
              {e.name} <span className={s.kindTag}>{e.kind === "bot" ? "bot" : "human"}</span>
            </td>
            <td className={s.rating}>{Math.round(e.rating)}</td>
            <td className={s.muted}>{e.games}</td>
            <td className={s.muted}>{winRate(e)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function LeaderboardView() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [bucket, setBucket] = useState<number | null>(null);
  const entries = useLeaderboard().data ?? [];

  useEffect(() => {
    void currentUser().then(setUser);
  }, []);

  // The player-count buckets that actually have ratings, ascending.
  const buckets = useMemo(
    () => [...new Set(entries.map((e) => e.n_players))].sort((a, b) => a - b),
    [entries]
  );
  const active = bucket ?? buckets[0] ?? null;
  const rows = useMemo(
    () => entries.filter((e) => e.n_players === active),
    [entries, active]
  );

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/" className={ui.backLink}>
        ‹ Menu
      </Link>

      <h1 className={s.title}>Leaderboard</h1>

      <Panel className={s.box}>
        <div className={s.tabs}>
          {buckets.map((n) => (
            <Button
              key={n}
              variant="small"
              selected={n === active}
              className={n === active ? s.accentText : undefined}
              onClick={() => setBucket(n)}
            >
              {n} players
            </Button>
          ))}
        </div>
        <Ladder rows={rows} />
      </Panel>
    </div>
  );
}
