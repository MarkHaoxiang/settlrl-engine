import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Button from "../components/Button";
import NewGameDialog from "../components/NewGameDialog";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import { matchmake, type PlayerCount } from "../lib/game";
import { useCreateGame } from "../lib/useCreateGame";
import { useLobby, type LobbyGame } from "../lib/queries";
import { rememberGame, saveTokens } from "../lib/seats";
import ui from "../styles/ui.module.css";
import s from "./LobbyView.module.css";

const POLL_MS = 2000;

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
  // The active Quick Match search, if any (its player count + how many wait).
  const [searching, setSearching] = useState<{ n: PlayerCount; waiting: number } | null>(null);
  const ticket = useRef<string | undefined>(undefined);
  const timer = useRef<number | null>(null);
  // Hosting a new game: the configuration dialog, plus the shared create flow
  // (handles the full-server queue and navigates into the game once it exists).
  const [hosting, setHosting] = useState(false);
  const { start: hostGame, queue, error: createError, cancel: abortCreate } = useCreateGame();

  useEffect(() => {
    void currentUser().then(setUser);
  }, []);
  // Stop polling if the view unmounts mid-search.
  useEffect(() => () => void (timer.current !== null && window.clearTimeout(timer.current)), []);

  // Joining is just a deep link: PlayView claims the first free seat on entry.
  const join = (id: string) => navigate(`/play/${id}`);

  // Re-POST the ticket on an interval until a seat comes back, then drop into it.
  const poll = async (n: PlayerCount) => {
    try {
      const res = await matchmake(n, ticket.current);
      if ("queued" in res) {
        ticket.current = res.ticket;
        setSearching({ n, waiting: res.waiting });
        timer.current = window.setTimeout(() => void poll(n), POLL_MS);
      } else {
        saveTokens(res.id, { [res.seat]: res.token });
        rememberGame(res.id);
        navigate(`/play/${res.id}`);
      }
    } catch {
      setSearching(null);
    }
  };
  const startMatch = (n: PlayerCount) => {
    ticket.current = undefined;
    setSearching({ n, waiting: 0 });
    void poll(n);
  };
  const cancelMatch = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    ticket.current = undefined;
    setSearching(null);
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

      <h1 className={s.title}>Lobby</h1>

      <Panel className={s.box}>
        <span className={ui.sectionLabel}>Host a game</span>
        <div className={s.quickRow}>
          <span className={s.muted}>Set up a board; list it here so others can join, or share the invite link.</span>
          <Button selected onClick={() => setHosting(true)}>
            Create game
          </Button>
        </div>
      </Panel>

      <Panel className={s.box}>
        <span className={ui.sectionLabel}>Quick Match</span>
        {searching ? (
          <div className={s.quickRow}>
            <span className={s.muted}>
              Finding a {searching.n}-player game… {searching.waiting} waiting
            </span>
            <Button onClick={cancelMatch}>Cancel</Button>
          </div>
        ) : (
          <div className={s.quickRow}>
            <span className={s.muted}>Pair with players near your rating; bots fill the rest.</span>
            <div className={s.quickButtons}>
              <Button selected onClick={() => startMatch(2)}>
                2 players
              </Button>
              <Button selected onClick={() => startMatch(4)}>
                4 players
              </Button>
            </div>
          </div>
        )}
      </Panel>

      <Panel className={s.box}>
        <span className={ui.sectionLabel}>Open games</span>
        {games.length === 0 ? (
          <span className={s.empty}>
            No open games right now. Host one above and list it in the lobby for others to join.
          </span>
        ) : (
          games.map((g) => <GameRow key={g.id} game={g} onJoin={join} />)
        )}
      </Panel>

      {createError && <div className={ui.overlayMsg}>{createError}</div>}
      {queue && (
        <div className={ui.overlayMsg}>
          You're #{queue.position} of {queue.total} in line…
          <div className={s.queueSub}>The server is busy; your game starts automatically.</div>
          <Button variant="small" className={s.queueCancel} onClick={abortCreate}>
            Cancel
          </Button>
        </div>
      )}
      {hosting && (
        <NewGameDialog
          onStart={(c) => {
            setHosting(false);
            hostGame(c);
          }}
          onClose={() => setHosting(false)}
        />
      )}
    </div>
  );
}
