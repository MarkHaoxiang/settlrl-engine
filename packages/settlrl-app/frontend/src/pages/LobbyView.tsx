import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import Button from "../components/Button";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { authToken, currentUser, type AuthUser } from "../lib/auth";
import { fetchGame, matchmake, type PlayerCount } from "../lib/game";
import { createLobby, getLobby, leaveLobby, type LobbyMode } from "../lib/lobby";
import { useLobbies, type LobbyListing } from "../lib/queries";
import {
  clearCurrentPlace,
  currentPlace,
  saveTokens,
  setCurrentPlace,
  tokensFor,
  type CurrentPlace,
} from "../lib/seats";
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

function LobbyRow({
  lobby,
  onJoin,
  disabled,
}: {
  lobby: LobbyListing;
  onJoin: (id: string) => void;
  disabled?: boolean;
}) {
  const seated = lobby.n_players - lobby.open_seats;
  return (
    <div className={s.row}>
      <div className={s.rowMain}>
        <span className={s.players}>
          {lobby.n_players} players
          {lobby.searchable && <span className={s.qmTag}>⚡ Quick Match</span>}
        </span>
        <span className={s.muted}>
          {seated}/{lobby.n_players} seated · {lobby.number_placement} map · {ago(lobby.created_at)}
        </span>
      </div>
      <Button
        selected
        disabled={disabled}
        title={disabled ? "Leave your current game first" : undefined}
        onClick={() => onJoin(lobby.id)}
      >
        Join
      </Button>
    </div>
  );
}

export default function LobbyView() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const navigate = useNavigate();
  const lobbies = useLobbies().data ?? [];
  // The place you're already in (a lobby or a live game); you can only be in one
  // at a time, so hosting / matching / joining another is gated until you leave.
  const [current, setCurrent] = useState<CurrentPlace | null>(currentPlace());
  const [error, setError] = useState<string | null>(null);
  // The active Quick Match search, if any (its player count + how many wait).
  const [searching, setSearching] = useState<{ n: PlayerCount; waiting: number } | null>(null);
  const ticket = useRef<string | undefined>(undefined);
  const timer = useRef<number | null>(null);
  const searchActive = useRef(false);

  useEffect(() => {
    void currentUser().then(setUser);
  }, []);

  // Verify the place we think we're in still exists, so a stale localStorage
  // entry (its game cleaned up, or finished) doesn't strand a guest behind a
  // phantom "you're in a game". Forget it if it's gone or over; follow a lobby
  // that has since started into its game.
  useEffect(() => {
    if (!current) return;
    let cancelled = false;
    const forget = () => {
      clearCurrentPlace(current.id);
      if (!cancelled) setCurrent(null);
    };
    void (async () => {
      try {
        if (current.kind === "game") {
          const g = await fetchGame(current.id, tokensFor(current.id));
          if (g.status.terminal) forget();
        } else {
          const lobby = await getLobby(current.id, tokensFor(current.id));
          if (lobby.started_game_id) {
            setCurrentPlace(lobby.started_game_id, "game");
            if (!cancelled) setCurrent({ id: lobby.started_game_id, kind: "game" });
          }
        }
      } catch {
        forget(); // 404 / cleaned up
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => () => void (timer.current !== null && window.clearTimeout(timer.current)), []);

  // Joining a listed lobby is a deep link; the lobby room claims a free seat.
  const join = (id: string) => navigate(`/lobby/${id}`);

  // Open a lobby — a local hotseat (you drive every seat) or an online table
  // (others join the open seats) — and land in its room to set it up and start.
  const host = (mode: LobbyMode) => {
    createLobby({ mode, listed: mode === "online" && !!authToken() }).then(
      (res) => {
        saveTokens(res.id, res.tokens);
        setCurrentPlace(res.id, "lobby");
        navigate(`/lobby/${res.id}`);
      },
      (e: unknown) => setError(`Could not open the lobby: ${String(e)}`)
    );
  };

  const poll = async (n: PlayerCount) => {
    try {
      const res = await matchmake(n, ticket.current);
      if ("queued" in res) {
        ticket.current = res.ticket;
        setSearching({ n, waiting: res.waiting });
        timer.current = window.setTimeout(() => void poll(n), POLL_MS);
      } else {
        searchActive.current = false;
        saveTokens(res.id, { [res.seat]: res.token });
        setCurrentPlace(res.id, "game");
        navigate(`/play/${res.id}`);
      }
    } catch {
      searchActive.current = false;
      setSearching(null);
    }
  };
  const startMatch = (n: PlayerCount) => {
    if (searchActive.current) return;
    searchActive.current = true;
    ticket.current = undefined;
    setSearching({ n, waiting: 0 });
    void poll(n);
  };
  const cancelMatch = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    ticket.current = undefined;
    searchActive.current = false;
    setSearching(null);
  };

  // Leave the lobby you're in so you're free to start another (a live game can't
  // be abandoned — only resumed). The host closes it; anyone else frees a seat.
  const leaveCurrent = async () => {
    if (!current || current.kind !== "lobby") return;
    try {
      await leaveLobby(current.id, tokensFor(current.id));
    } catch {
      // Already gone — either way stop tracking it.
    }
    clearCurrentPlace(current.id);
    setCurrent(null);
  };

  const gated = !!current;

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/" className={ui.backLink}>
        ‹ Menu
      </Link>

      <h1 className={s.title}>Play</h1>

      {current && (
        <Panel className={s.box}>
          <span className={ui.sectionLabel}>You're in a {current.kind === "lobby" ? "lobby" : "game"}</span>
          <div className={s.quickRow}>
            <span className={s.muted}>
              You can only be in one at a time. Resume it
              {current.kind === "lobby" ? ", or leave to start another." : "."}
            </span>
            <div className={s.quickButtons}>
              <Button
                selected
                onClick={() =>
                  navigate(current.kind === "lobby" ? `/lobby/${current.id}` : `/play/${current.id}`)
                }
              >
                Resume
              </Button>
              {current.kind === "lobby" && (
                <Button onClick={() => void leaveCurrent()}>Leave</Button>
              )}
            </div>
          </div>
        </Panel>
      )}

      <Panel className={s.box}>
        <span className={ui.sectionLabel}>Host a game</span>
        <div className={s.quickRow}>
          <span className={s.muted}>
            Local hotseat (pass-and-play on this device) or an online table others can join.
          </span>
          <div className={s.quickButtons}>
            <Button
              disabled={gated}
              title={gated ? "Leave your current game first" : undefined}
              onClick={() => host("hotseat")}
            >
              Local hotseat
            </Button>
            <Button
              selected
              disabled={gated}
              title={gated ? "Leave your current game first" : undefined}
              onClick={() => host("online")}
            >
              Online table
            </Button>
          </div>
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
              <Button
                selected
                disabled={gated}
                title={gated ? "Leave your current game first" : undefined}
                onClick={() => startMatch(2)}
              >
                2 players
              </Button>
              <Button
                selected
                disabled={gated}
                title={gated ? "Leave your current game first" : undefined}
                onClick={() => startMatch(4)}
              >
                4 players
              </Button>
            </div>
          </div>
        )}
      </Panel>

      <Panel className={s.box}>
        <span className={ui.sectionLabel}>Open tables</span>
        {lobbies.length === 0 ? (
          <span className={s.empty}>
            No open tables right now. Host one above and list it for others to join.
          </span>
        ) : (
          lobbies.map((l) => <LobbyRow key={l.id} lobby={l} onJoin={join} disabled={gated} />)
        )}
      </Panel>

      {error && <div className={ui.overlayMsg}>{error}</div>}
    </div>
  );
}
