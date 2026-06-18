import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import { useLeaderboard, type LeaderboardEntry } from "../lib/queries";
import { ACCENT, DIVIDER, LINK, panelStyle, selectedStyle, smallButtonStyle } from "../lib/ui";

const winRate = (e: LeaderboardEntry) =>
  e.games ? `${Math.round((100 * e.wins) / e.games)}%` : "—";

function Ladder({ rows }: { rows: LeaderboardEntry[] }) {
  if (rows.length === 0)
    return <span style={{ fontSize: 13, opacity: 0.6 }}>No rated games yet.</span>;
  return (
    <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 14 }}>
      <thead>
        <tr style={{ textAlign: "left", opacity: 0.6, fontSize: 12 }}>
          <th style={{ padding: "6px 10px" }}>#</th>
          <th style={{ padding: "6px 10px" }}>Player</th>
          <th style={{ padding: "6px 10px", textAlign: "right" }}>Rating</th>
          <th style={{ padding: "6px 10px", textAlign: "right" }}>Games</th>
          <th style={{ padding: "6px 10px", textAlign: "right" }}>Win</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((e, i) => (
          <tr key={`${e.kind}:${e.name}`} style={{ borderTop: `1px solid ${DIVIDER}` }}>
            <td style={{ padding: "6px 10px", opacity: 0.6 }}>{i + 1}</td>
            <td style={{ padding: "6px 10px" }}>
              {e.name}{" "}
              <span
                style={{
                  fontSize: 10,
                  opacity: 0.6,
                  textTransform: "uppercase",
                  letterSpacing: 0.5,
                }}
              >
                {e.kind === "bot" ? "bot" : "human"}
              </span>
            </td>
            <td style={{ padding: "6px 10px", textAlign: "right", fontWeight: 700 }}>
              {Math.round(e.rating)}
            </td>
            <td style={{ padding: "6px 10px", textAlign: "right", opacity: 0.75 }}>
              {e.games}
            </td>
            <td style={{ padding: "6px 10px", textAlign: "right", opacity: 0.75 }}>
              {winRate(e)}
            </td>
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
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 28,
        padding: 24,
        color: "var(--text)",
        fontFamily: "Georgia, serif",
      }}
    >
      <div style={{ position: "fixed", top: 16, right: 16, display: "flex", gap: 10 }}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/" style={{ position: "fixed", top: 16, left: 16, color: LINK }}>
        ‹ Menu
      </Link>

      <h1 style={{ fontSize: 36, margin: 0 }}>Leaderboard</h1>

      <div
        style={{ ...panelStyle, padding: "16px 20px", borderRadius: 12, minWidth: 420 }}
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          {buckets.map((n) => (
            <button
              key={n}
              onClick={() => setBucket(n)}
              style={{
                ...smallButtonStyle,
                ...(n === active ? selectedStyle : {}),
                color: n === active ? ACCENT : "var(--text)",
              }}
            >
              {n} players
            </button>
          ))}
        </div>
        <Ladder rows={rows} />
      </div>
    </div>
  );
}
