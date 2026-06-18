import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import MyGames from "../components/MyGames";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import { downloadRecord } from "../lib/game";
import { useHistory, type PastGame } from "../lib/queries";
import { loadReplayFromGame } from "../lib/replay";
import { LINK, buttonStyle, panelStyle, smallButtonStyle } from "../lib/ui";

const outcome = (g: PastGame) =>
  g.winner == null
    ? "Game over"
    : g.seats.includes(g.winner)
      ? "You won 🎉"
      : `Player ${g.winner + 1} won`;

export default function ProfileView() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [checked, setChecked] = useState(false);
  const past = useHistory(user).data ?? [];
  const navigate = useNavigate();

  useEffect(() => {
    void currentUser().then((u) => {
      setUser(u);
      setChecked(true);
    });
  }, []);

  const replay = async (id: string) => {
    await loadReplayFromGame(id);
    navigate("/replay");
  };

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

      <h1 style={{ fontSize: 36, margin: 0 }}>Profile</h1>

      {checked && !user && (
        <div style={{ ...panelStyle, padding: "20px 24px", borderRadius: 12 }}>
          <Link to="/login" style={{ color: LINK }}>
            Sign in
          </Link>{" "}
          to see your games.
        </div>
      )}

      {user && <MyGames user={user} />}

      {user && (
        <div style={{ ...panelStyle, padding: "16px 20px", borderRadius: 12, minWidth: 360 }}>
          <div
            style={{
              fontSize: 12,
              opacity: 0.6,
              textTransform: "uppercase",
              letterSpacing: 1,
              marginBottom: 10,
            }}
          >
            Past games
          </div>
          {past.length === 0 ? (
            <span style={{ fontSize: 13, opacity: 0.6 }}>No finished games yet.</span>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {past.map((g) => (
                <div
                  key={g.id}
                  style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}
                >
                  <span style={{ flex: 1 }}>
                    <b>{outcome(g)}</b>{" "}
                    <span style={{ opacity: 0.6 }}>
                      · {g.n_players}p · seat{g.seats.length > 1 ? "s" : ""}{" "}
                      {g.seats.map((s) => s + 1).join(", ")}
                    </span>
                    <br />
                    <span style={{ opacity: 0.5, fontSize: 11 }}>
                      {new Date(g.finished_at * 1000).toLocaleString()}
                    </span>
                  </span>
                  <button style={smallButtonStyle} onClick={() => void replay(g.id)}>
                    Replay
                  </button>
                  <button style={smallButtonStyle} onClick={() => void downloadRecord(g.id)}>
                    Download
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <Link to="/play" style={{ ...buttonStyle, color: LINK, textDecoration: "none" }}>
        New game
      </Link>
    </div>
  );
}
