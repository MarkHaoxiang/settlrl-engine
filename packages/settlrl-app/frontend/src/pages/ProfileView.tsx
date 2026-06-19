import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Button from "../components/Button";
import MyGames from "../components/MyGames";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import { downloadRecord } from "../lib/game";
import { useHistory, type PastGame } from "../lib/queries";
import { loadReplayFromGame } from "../lib/replay";
import ui from "../styles/ui.module.css";
import s from "./ProfileView.module.css";

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
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/" className={ui.backLink}>
        ‹ Menu
      </Link>

      <h1 className={s.title}>Profile</h1>

      {checked && !user && (
        <Panel className={s.signinBox}>
          <Link to="/login" className={ui.link}>
            Sign in
          </Link>{" "}
          to see your games.
        </Panel>
      )}

      {user && <MyGames user={user} />}

      {user && (
        <Panel className={s.pastBox}>
          <div className={s.label}>Past games</div>
          {past.length === 0 ? (
            <span className={s.empty}>No finished games yet.</span>
          ) : (
            <div className={s.list}>
              {past.map((g) => (
                <div key={g.id} className={s.gameRow}>
                  <span className={s.gameInfo}>
                    <b>{outcome(g)}</b>{" "}
                    <span className={s.meta}>
                      · {g.n_players}p · seat{g.seats.length > 1 ? "s" : ""}{" "}
                      {g.seats.map((s) => s + 1).join(", ")}
                    </span>
                    <br />
                    <span className={s.date}>
                      {new Date(g.finished_at * 1000).toLocaleString()}
                    </span>
                  </span>
                  <Button variant="small" onClick={() => void replay(g.id)}>
                    Replay
                  </Button>
                  <Button variant="small" onClick={() => void downloadRecord(g.id)}>
                    Download
                  </Button>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}

      <Link to="/play" className={ui.buttonLink}>
        New game
      </Link>
    </div>
  );
}
