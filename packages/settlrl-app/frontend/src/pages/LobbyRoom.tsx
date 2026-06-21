// The lobby room for one game (/lobby/:id): a full-page, three-column staging
// area — Players (left), Map & settings (middle), Chat (right). The game already
// exists, so the room renders the live board straight from the SSE snapshot and
// the host's edits reconfigure it in place (every participant sees them). When
// the table fills (every human seat claimed) everyone is sent into /play/:id.

import { useEffect, useMemo, useRef, useState } from "react";
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
import {
  configureGame,
  fetchBots,
  joinGame,
  leaveGame,
  setSeat,
  type BotSpec,
  type GameConfig,
  type NumberPlacement,
  type PlayerCount,
} from "../lib/game";
import {
  clearCurrentGame,
  saveTokens,
  setCurrentGame,
  tokensFor,
  type SeatTokens,
} from "../lib/seats";
import { useGame } from "../lib/useGame";
import ui from "../styles/ui.module.css";
import s from "./LobbyRoom.module.css";

const botLabel = (kind: string, spec?: BotSpec) =>
  spec?.title ?? (kind === "mcts" ? "MCTS" : kind.charAt(0).toUpperCase() + kind.slice(1));

export default function LobbyRoom() {
  const { id: gameId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [tokens, setTokens] = useState<SeatTokens>(() => (gameId ? tokensFor(gameId) : {}));
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
    if (gameId) setTokens(tokensFor(gameId));
    joining.current = false;
    setJoinFailed(false);
  }, [gameId]);

  const { snapshot, error, chat } = useGame(gameId ?? null, tokens);

  // An invitee arriving with no seat takes the first free human seat (else
  // spectates). Signed-in players already owning a seat here are skipped.
  useEffect(() => {
    if (!gameId || joinFailed || joining.current || Object.keys(tokens).length > 0) return;
    if (Object.keys(tokensFor(gameId)).length > 0) return;
    if (authToken()) {
      if (!snapshot) return;
      if (snapshot.your_seats.length > 0) return;
    }
    joining.current = true;
    joinGame(gameId).then(
      (j) => {
        saveTokens(gameId, { [j.seat]: j.token });
        setCurrentGame(gameId);
        joining.current = false;
        setTokens(tokensFor(gameId));
      },
      () => {
        joining.current = false;
        setJoinFailed(true);
      }
    );
  }, [gameId, tokens, joinFailed, snapshot]);

  const status = snapshot?.status;
  const seatKinds = status?.seats ?? [];
  const claimed = useMemo(() => new Set(snapshot?.seats_claimed ?? []), [snapshot]);
  const humanSeats = seatKinds.flatMap((k, i) => (k === "human" ? [i] : []));
  const waiting = !!status && !status.terminal && humanSeats.some((i) => !claimed.has(i));

  // The table is full (or the game's over): play has begun — drop into it.
  useEffect(() => {
    if (gameId && snapshot && !waiting) navigate(`/play/${gameId}`, { replace: true });
  }, [gameId, snapshot, waiting, navigate]);

  // The game vanished (the host closed the lobby, or it was evicted): the stream
  // 404s — stop tracking it and bounce back to the lobby list.
  useEffect(() => {
    if (error) {
      clearCurrentGame(gameId ?? undefined);
      navigate("/lobby", { replace: true });
    }
  }, [error, gameId, navigate]);

  if (error) return <div className={ui.overlayMsg}>{error}</div>;
  if (!snapshot || !status || !gameId) return <div className={ui.overlayMsg}>Loading lobby…</div>;

  const mySeats = snapshot.your_seats;
  const isHost = mySeats.includes(0);
  const n = seatKinds.length as PlayerCount;
  const placement = (snapshot.number_placement as NumberPlacement) ?? "random";
  const signedIn = !!authToken();

  const botNames = Object.keys(bots)
    .filter((b) => bots[b].counts.includes(n))
    .sort();
  const defaultBot = botNames.includes("random") ? "random" : (botNames[0] ?? "random");

  // Leave before the game starts: the host (seat 0) closes the whole lobby and
  // everyone else is bounced; any other participant just frees their seat.
  const leave = () =>
    leaveGame(gameId, tokens).then(
      () => {
        clearCurrentGame(gameId);
        navigate("/lobby");
      },
      (e) => setMsg(String(e))
    );

  const reconfigure = (cfg: GameConfig) =>
    configureGame(gameId, tokens, cfg).catch((e) => setMsg(String(e)));
  const retarget = (seat: number, kind: string) =>
    setSeat(gameId, tokens, seat, kind).catch((e) => setMsg(String(e)));

  const seatLabel = (i: number): string =>
    seatKinds[i] !== "human"
      ? botLabel(seatKinds[i], bots[seatKinds[i]])
      : claimed.has(i)
        ? (snapshot.seat_names[i] ?? "Guest") + (mySeats.includes(i) ? " (you)" : "")
        : "open";

  // Bot-fill every still-open human seat so a vs-bots game can start now; the
  // table then becomes full and the redirect above carries everyone into play.
  const startGame = () => {
    for (const i of humanSeats) if (!claimed.has(i)) void retarget(i, defaultBot);
  };

  const copyInvite = () => {
    void navigator.clipboard.writeText(window.location.href);
    setLinkCopied(true);
    window.setTimeout(() => setLinkCopied(false), 1500);
  };

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <Link to="/lobby" className={ui.backLink}>
        ‹ Lobby
      </Link>
      <h1 className={s.title}>Lobby</h1>

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
            {seatKinds.map((kind, i) => {
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
                        title="Open this seat for a human player"
                        aria-label="Open this seat for a human player"
                        onClick={() => retarget(i, "human")}
                      >
                        <HumanIcon size={16} />
                      </button>
                      {/* Only offer a bot when the server actually serves one; the
                          icon opens the catalog overlay to pick which. */}
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
                  value={status.victory_points_to_win}
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
                {n} players · {placement} map · first to {status.victory_points_to_win} VP
              </span>
            </div>
          )}

          {isHost && (
            <div className={s.flags}>
              <button
                className={cx(s.flagBtn, snapshot.listed && ui.selected)}
                disabled={!signedIn}
                title={signedIn ? "Show this game in the public lobby" : "Sign in to list a game"}
                onClick={() => reconfigure({ listed: !snapshot.listed })}
              >
                {snapshot.listed ? "Listed" : "List in lobby"}
              </button>
              <button
                className={cx(s.flagBtn, snapshot.searchable && ui.selected)}
                disabled={!signedIn}
                title={signedIn ? "Open this game to Quick Match" : "Sign in to enable Quick Match"}
                onClick={() => reconfigure({ searchable: !snapshot.searchable })}
              >
                {snapshot.searchable ? "Searchable" : "Open to Quick Match"}
              </button>
            </div>
          )}

          <div className={s.actions}>
            <Button onClick={copyInvite} title="Others join the open seats by opening this link">
              {linkCopied ? "Copied!" : "🔗 Invite link"}
            </Button>
            <Button
              onClick={() => void leave()}
              title={isHost ? "Close this lobby for everyone" : "Leave this lobby"}
            >
              {isHost ? "Close lobby" : "Leave"}
            </Button>
            {isHost && botNames.length > 0 && (
              <Button selected onClick={startGame} title="Fill open seats with bots and start">
                Start game
              </Button>
            )}
          </div>
          <span className={s.dim}>
            {humanSeats.filter((i) => claimed.has(i)).length} / {humanSeats.length} seats filled —
            {isHost ? " start now or wait for players to join." : " waiting for the host to start."}
          </span>
        </Panel>

        {/* Chat */}
        <div className={s.chat}>
          <ChatPanel
            entries={snapshot.log}
            onSend={(text) => chat(text, mySeats[0] ?? null)}
            players={snapshot.board.players}
            you={mySeats[0]}
            identities={seatKinds.map((_, i) => seatLabel(i))}
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
                const current = seatKinds[picker] === b;
                return (
                  <button
                    key={b}
                    className={cx(s.pickerItem, current && ui.selected)}
                    onClick={() => {
                      void retarget(picker, b);
                      setPicker(null);
                    }}
                  >
                    <BotIcon size={20} />
                    <span className={s.pickerName}>{botLabel(b, bots[b])}</span>
                    <span className={s.pickerDesc}>{bots[b]?.description}</span>
                    {current && <span className={s.pickerCheck}>✓</span>}
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
