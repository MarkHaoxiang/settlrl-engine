// The lobby room for one table (/lobby/:id): a full-page, three-column staging
// area — Players (left), Map & settings (middle), Chat (right). A lobby holds
// only configuration and claimed seats (no engine); the board is a preview from
// the seed. The host edits it live (every participant sees it) and Starts only
// once every seat is decided, at which point it materialises into a game and
// everyone is sent into /play/:id.

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import BoardView from "../components/BoardView";
import Button from "../components/Button";
import ChatPanel from "../components/ChatPanel";
import Modal from "../components/Modal";
import Panel from "../components/Panel";
import ThemeToggle from "../components/ThemeToggle";
import { BotIcon, HumanIcon, MapIcon } from "../components/icons";
import { authToken, currentUser, type AuthUser } from "../lib/auth";
import { PLAYER_COLORS, playerName } from "../lib/boardData";
import { cx } from "../lib/cx";
import { fetchBots, type BotSpec, type LogEntry } from "../lib/game";
import {
  configureLobby,
  joinLobby,
  leaveLobby,
  setLobbySeat,
  startLobby,
} from "../lib/lobby";
import {
  clearCurrentPlace,
  saveTokens,
  setCurrentPlace,
  tokensFor,
  type SeatTokens,
} from "../lib/seats";
import { useLobby } from "../lib/useLobby";
import ui from "../styles/ui.module.css";
import s from "./LobbyRoom.module.css";

const botLabel = (kind: string, spec?: BotSpec) =>
  spec?.title ?? (kind === "mcts" ? "MCTS" : kind.charAt(0).toUpperCase() + kind.slice(1));

export default function LobbyRoom() {
  const { id: lobbyId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [tokens, setTokens] = useState<SeatTokens>(() => (lobbyId ? tokensFor(lobbyId) : {}));
  const [joinFailed, setJoinFailed] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [linkCopied, setLinkCopied] = useState(false);
  // The seat whose bot-picker overlay is open (null = closed).
  const [picker, setPicker] = useState<number | null>(null);
  const joining = useRef(false);
  const [bots, setBots] = useState<Record<string, BotSpec>>({});

  useEffect(() => {
    void currentUser().then(setUser);
    fetchBots().then(setBots).catch(() => setBots({}));
  }, []);

  useEffect(() => {
    if (lobbyId) setTokens(tokensFor(lobbyId));
    joining.current = false;
    setJoinFailed(false);
  }, [lobbyId]);

  const { snapshot, error, chat } = useLobby(lobbyId ?? null, tokens);

  // An invitee arriving at an online table with no seat takes the first open one
  // (else spectates). A hotseat has no open seats, so a visitor just spectates.
  useEffect(() => {
    if (!lobbyId || joinFailed || joining.current || Object.keys(tokens).length > 0) return;
    if (Object.keys(tokensFor(lobbyId)).length > 0) return;
    if (!snapshot || snapshot.mode !== "online" || snapshot.your_seats.length > 0) return;
    if (snapshot.seats_claimed.length >= snapshot.n_players) return;
    joining.current = true;
    joinLobby(lobbyId).then(
      (j) => {
        saveTokens(lobbyId, j.tokens);
        setCurrentPlace(lobbyId, "lobby");
        joining.current = false;
        setTokens(tokensFor(lobbyId));
      },
      () => {
        joining.current = false;
        setJoinFailed(true);
      }
    );
  }, [lobbyId, tokens, joinFailed, snapshot]);

  // Someone started the table: follow everyone into the game.
  useEffect(() => {
    if (snapshot?.started_game_id) {
      setCurrentPlace(snapshot.started_game_id, "game");
      navigate(`/play/${snapshot.started_game_id}`, { replace: true });
    }
  }, [snapshot?.started_game_id, navigate]);

  // The lobby vanished (the host closed it, or it was evicted): stop tracking it
  // and bounce back to the lobby list.
  useEffect(() => {
    if (error) {
      clearCurrentPlace(lobbyId ?? undefined);
      navigate("/lobby", { replace: true });
    }
  }, [error, lobbyId, navigate]);

  if (error) return <div className={ui.overlayMsg}>{error}</div>;
  if (!snapshot || !lobbyId) return <div className={ui.overlayMsg}>Loading lobby…</div>;

  const { kinds, n_players: n, mode } = snapshot;
  const mySeats = snapshot.your_seats;
  const isHost = mySeats.includes(0);
  const isOnline = mode === "online";
  const claimed = new Set(snapshot.seats_claimed);
  const placement = snapshot.number_placement;
  const signedIn = !!authToken();

  const botNames = Object.keys(bots)
    .filter((b) => bots[b].counts.includes(n))
    .sort();

  const reconfigure = (cfg: Parameters<typeof configureLobby>[2]) =>
    configureLobby(lobbyId, tokens, cfg).catch((e) => setMsg(String(e)));
  const retarget = (seat: number, kind: string) =>
    setLobbySeat(lobbyId, tokens, seat, kind).catch((e) => setMsg(String(e)));

  const seatLabel = (i: number): string =>
    kinds[i] !== "human"
      ? botLabel(kinds[i], bots[kinds[i]])
      : claimed.has(i)
        ? (snapshot.seat_names[i] ?? "Guest") + (mySeats.includes(i) ? " (you)" : "")
        : "open";

  // Materialise the table into a game (host only; enabled once every seat is
  // decided). The host jumps straight in; everyone else follows via the SSE.
  const start = () =>
    startLobby(lobbyId, tokens).then((res) => {
      if ("game_id" in res) {
        setCurrentPlace(res.game_id, "game");
        navigate(`/play/${res.game_id}`, { replace: true });
      } else {
        setMsg(`Server busy — you're #${res.position} in line; starting soon.`);
      }
    }, (e) => setMsg(String(e)));

  const leave = () =>
    leaveLobby(lobbyId, tokens).then(() => {
      clearCurrentPlace(lobbyId);
      navigate("/lobby");
    }, (e) => setMsg(String(e)));

  const copyInvite = () => {
    void navigator.clipboard.writeText(window.location.href);
    setLinkCopied(true);
    window.setTimeout(() => setLinkCopied(false), 1500);
  };

  const log = snapshot.chat.map((c, i) => ({
    id: i,
    kind: "chat",
    player: (c.player ?? null) as number | null,
    text: String(c.text),
  })) as unknown as LogEntry[];

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/lobby" className={ui.backLink}>
        ‹ Lobby
      </Link>
      <h1 className={s.title}>{isOnline ? "Online table" : "Local hotseat"}</h1>

      <div className={s.columns}>
        {/* Players */}
        <Panel className={s.col}>
          <span className={ui.sectionLabel}>Players</span>
          {isHost && (
            <div className={s.field}>
              <span className={s.fieldLabel}>Count</span>
              <div className={s.toggle}>
                {([2, 4] as const).map((c) => (
                  <button
                    key={c}
                    className={cx(s.toggleBtn, n === c && ui.selected)}
                    onClick={() => reconfigure({ nPlayers: c })}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className={s.seatList}>
            {kinds.map((kind, i) => {
              const human = kind === "human";
              const open = human && !claimed.has(i);
              return (
                <div key={i} className={s.seatRow} style={{ opacity: open ? 0.6 : 1 }}>
                  <span className={s.dot} style={{ background: PLAYER_COLORS[i] ?? "#888" }} />
                  {human ? <HumanIcon size={16} /> : <BotIcon size={16} />}
                  <span className={s.seatName}>{playerName(i)}</span>
                  <span className={s.seatStatus}>{seatLabel(i)}</span>
                  {/* The host retargets only seats no one has claimed yet. */}
                  {isHost && !claimed.has(i) && (
                    <div className={s.seatControls}>
                      <button
                        className={cx(s.iconBtn, human && ui.selected)}
                        title={isOnline ? "Open this seat for a human player" : "Take this seat yourself"}
                        aria-label="Human seat"
                        onClick={() => retarget(i, "human")}
                      >
                        <HumanIcon size={16} />
                      </button>
                      {botNames.length > 0 && (
                        <button
                          className={cx(s.iconBtn, !human && ui.selected)}
                          title="Fill this seat with a bot"
                          aria-label="Fill this seat with a bot"
                          onClick={() => setPicker(i)}
                        >
                          <BotIcon size={16} />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Panel>

        {/* Map & settings */}
        <Panel className={s.col}>
          <span className={ui.sectionLabel}>Map &amp; settings</span>
          <div className={s.board}>
            <BoardView board={snapshot.board} />
          </div>
          {isHost ? (
            <>
              <div className={s.field}>
                <span className={s.fieldLabel}>Numbers</span>
                <div className={s.toggle}>
                  {(["random", "spiral"] as const).map((p) => (
                    <button
                      key={p}
                      className={cx(s.toggleBtn, placement === p && ui.selected)}
                      onClick={() =>
                        reconfigure({ numberPlacement: p, seed: Math.floor(Math.random() * 65536) })
                      }
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
              <div className={s.field}>
                <span className={s.fieldLabel}>Map</span>
                <button
                  className={s.mapBtn}
                  title="Roll a new random map"
                  onClick={() => reconfigure({ seed: Math.floor(Math.random() * 65536) })}
                >
                  <MapIcon /> 🎲 <span className={s.dim}>#{snapshot.seed}</span>
                </button>
              </div>
              <div className={s.field}>
                <span className={s.fieldLabel}>Win at</span>
                <input
                  type="number"
                  className={s.vpInput}
                  min={3}
                  max={20}
                  value={snapshot.victory_points_to_win}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v) && v >= 3) reconfigure({ victoryPointsToWin: Math.round(v) });
                  }}
                  title="Victory points needed to win"
                />
                <span className={s.dim}>VP</span>
              </div>
            </>
          ) : (
            <div className={s.field}>
              <span className={s.dim}>
                {n} players · {placement} map · first to {snapshot.victory_points_to_win} VP
              </span>
            </div>
          )}

          {isHost && isOnline && (
            <div className={s.flags}>
              <button
                className={cx(s.flagBtn, snapshot.listed && ui.selected)}
                disabled={!signedIn}
                title={signedIn ? "Show this table in the public list" : "Sign in to list a table"}
                onClick={() => reconfigure({ listed: !snapshot.listed })}
              >
                {snapshot.listed ? "Listed" : "List publicly"}
              </button>
              <button
                className={cx(s.flagBtn, snapshot.searchable && ui.selected)}
                disabled={!signedIn}
                title={signedIn ? "Open this table to Quick Match" : "Sign in to enable Quick Match"}
                onClick={() => reconfigure({ searchable: !snapshot.searchable })}
              >
                {snapshot.searchable ? "Searchable" : "Open to Quick Match"}
              </button>
            </div>
          )}

          <div className={s.actions}>
            {isOnline && (
              <Button onClick={copyInvite} title="Others join the open seats by opening this link">
                {linkCopied ? "Copied!" : "🔗 Invite link"}
              </Button>
            )}
            <Button
              onClick={() => void leave()}
              title={isHost ? "Close this lobby for everyone" : "Leave this lobby"}
            >
              {isHost ? "Close lobby" : "Leave"}
            </Button>
            {isHost && (
              <Button
                selected
                disabled={!snapshot.ready}
                onClick={() => void start()}
                title={snapshot.ready ? "Start the game" : "Fill every open seat first"}
              >
                Start game
              </Button>
            )}
          </div>
          <span className={s.dim}>
            {snapshot.seats_claimed.length} / {n} seats taken —
            {snapshot.ready
              ? " ready to start."
              : isHost
                ? " fill the open seats with players or bots."
                : " waiting for the host to start."}
          </span>
        </Panel>

        {/* Chat */}
        <div className={s.chat}>
          <ChatPanel
            entries={log}
            onSend={(text) => chat(text, mySeats[0] ?? null)}
            players={snapshot.board.players}
            you={mySeats[0]}
            identities={kinds.map((_, i) => seatLabel(i))}
          />
        </div>
      </div>

      {picker !== null && (
        <Modal onClose={() => setPicker(null)} title="Choose a bot">
          <div className={cx(ui.panel, s.picker)}>
            <div className={s.pickerHead}>
              <span className={ui.sectionLabel}>Choose a bot</span>
              <button className={s.pickerClose} aria-label="Close" onClick={() => setPicker(null)}>
                ✕
              </button>
            </div>
            <div className={s.pickerList}>
              {botNames.map((b) => {
                const here = kinds[picker] === b;
                return (
                  <button
                    key={b}
                    className={cx(s.pickerItem, here && ui.selected)}
                    onClick={() => {
                      void retarget(picker, b);
                      setPicker(null);
                    }}
                  >
                    <BotIcon size={20} />
                    <span className={s.pickerName}>{botLabel(b, bots[b])}</span>
                    <span className={s.pickerDesc}>{bots[b]?.description}</span>
                    {here && <span className={s.pickerCheck}>✓</span>}
                  </button>
                );
              })}
            </div>
          </div>
        </Modal>
      )}

      {msg && <div className={ui.overlayMsg}>{msg}</div>}
    </div>
  );
}
