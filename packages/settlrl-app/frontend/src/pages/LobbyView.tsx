import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Button from "../components/Button";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import { useLobby, type LobbyGame } from "../lib/queries";
import ui from "../styles/ui.module.css";
import s from "./LobbyView.module.css";

const ago = (epoch: number): string => {
  const secs = Math.max(0, Math.round(Date.now() / 1000 - epoch));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.round(mins / 60)}h ago`;
};

function GameRow({ game, onJoin }: { game: LobbyGame; onJoin: (id: string) => void }) {
  const seated = game.n_players - game.open_seats;
  return (
    <div className={s.row}>
      <div className={s.rowMain}>
        <span className={s.players}>{game.n_players} players</span>
        <span className={s.muted}>
          {seated}/{game.n_players} seated · {game.number_placement} map · {ago(game.created_at)}
        </span>
      </div>
      <Button selected onClick={() => onJoin(game.id)}>
        Join
      </Button>
    </div>
  );
}

export default function LobbyView() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const navigate = useNavigate();
  const games = useLobby().data ?? [];

  useEffect(() => {
    void currentUser().then(setUser);
  }, []);

  // Joining is just a deep link: PlayView claims the first free seat on entry.
  const join = (id: string) => navigate(`/play/${id}`);

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/" className={ui.backLink}>
        ‹ Menu
      </Link>

      <h1 className={s.title}>Lobby</h1>

      <Panel className={s.box}>
        {games.length === 0 ? (
          <span className={s.empty}>
            No open games right now. Create one and list it in the lobby for others to join.
          </span>
        ) : (
          games.map((g) => <GameRow key={g.id} game={g} onJoin={join} />)
        )}
      </Panel>
    </div>
  );
}
