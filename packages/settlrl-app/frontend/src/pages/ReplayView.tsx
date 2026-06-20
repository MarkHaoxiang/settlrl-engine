import { useEffect, useRef, useState } from "react";
import BoardView from "../components/BoardView";
import ChatPanel from "../components/ChatPanel";
import TopBar from "../components/TopBar";
import { actionMeta } from "../lib/actionMeta";
import { PLAYER_COLORS, playerName } from "../lib/boardData";
import {
  fetchReplayState,
  loadReplay,
  loadReplayFromGame,
  type ReplayState,
} from "../lib/replay";
import { lastGameId } from "../lib/seats";
import ui from "../styles/ui.module.css";
import s from "./ReplayView.module.css";

const PLAY_INTERVAL_MS = 600;

// The pair of ways to load a record: a saved .json file, or the most recent
// game from this browser. Rendered both on the empty state and (small) in the bar.
function LoadButtons({
  small,
  onLoad,
  onError,
}: {
  small?: boolean;
  onLoad: (s: ReplayState) => void;
  onError: (msg: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const cn = small ? s.ctrlWide : ui.button;

  const openFile = async (file: File) => {
    try {
      onLoad(await loadReplay(JSON.parse(await file.text())));
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = ""; // allow re-picking the same file
          if (file) void openFile(file);
        }}
      />
      <button className={cn} title="Open a saved record file" onClick={() => fileRef.current?.click()}>
        {small ? "📂" : "📂 Open record file…"}
      </button>
      <button
        className={cn}
        title="Replay your most recent game on this browser, as played so far"
        onClick={() => {
          const last = lastGameId();
          if (!last) onError("no recent game on this browser");
          else loadReplayFromGame(last).then(onLoad, (e: unknown) => onError(String(e)));
        }}
      >
        {small ? "🎮" : "🎮 Replay your last game"}
      </button>
    </>
  );
}

export default function ReplayView() {
  const [state, setState] = useState<ReplayState | null>(null);
  const [checked, setChecked] = useState(false); // initial server probe done
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0); // drop out-of-order seek responses

  // Pick up a replay already loaded on the server (e.g. after a refresh).
  useEffect(() => {
    let cancelled = false;
    fetchReplayState(0).then(
      (s) => {
        if (!cancelled) {
          setState(s);
          setChecked(true);
        }
      },
      () => !cancelled && setChecked(true) // none loaded yet
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const apply = (p: Promise<ReplayState>) => {
    const n = ++seq.current;
    p.then(
      (s) => {
        if (seq.current === n) {
          setState(s);
          setError(null);
        }
      },
      (e: unknown) => {
        if (seq.current === n) setError(e instanceof Error ? e.message : String(e));
      }
    );
  };

  const seek = (move: number) => {
    if (!state) return;
    apply(fetchReplayState(Math.max(0, Math.min(state.n_moves, move))));
  };

  const loaded = (s: ReplayState) => {
    setPlaying(false);
    setError(null);
    seq.current++;
    setState(s);
  };

  // Playback: each arriving state schedules the next step until the end.
  useEffect(() => {
    if (!playing || !state) return;
    if (state.move >= state.n_moves) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => apply(fetchReplayState(state.move + 1)), PLAY_INTERVAL_MS);
    return () => clearTimeout(t);
  }, [playing, state]);

  if (!checked) return null;

  // No record loaded yet: offer the two ways to get one.
  if (!state) {
    return (
      <div className={s.page}>
        <TopBar mode="Replay" />
        <div className={s.loadDialog}>
          <span className={s.loadTitle}>Replay a game</span>
          <span className={s.loadSub}>
            Open a saved game record, or replay your most recent game from its first move.
          </span>
          <div className={s.loadRow}>
            <LoadButtons onLoad={loaded} onError={setError} />
          </div>
          {error && <span className={s.error}>{error}</span>}
        </div>
      </div>
    );
  }

  const atEnd = state.move >= state.n_moves;
  const last = state.move > 0 ? state.log[state.log.length - 1] : undefined;

  return (
    <div className={s.layout}>
      <div className={s.boardArea}>
        <BoardView board={state.board} />
        <TopBar mode="Replay" />

        {atEnd && state.winner != null && (
          <div className={s.winnerBanner}>{playerName(state.winner)} wins</div>
        )}

        {/* Bottom centre: playback controls */}
        <div className={s.bottomStrip}>
          <div className={s.bar}>
            <button className={s.ctrl} title="Back to start" disabled={state.move === 0} onClick={() => seek(0)}>
              ⏮
            </button>
            <button className={s.ctrl} title="Step back" disabled={state.move === 0} onClick={() => seek(state.move - 1)}>
              ◀
            </button>
            <button
              className={s.ctrl}
              title={playing ? "Pause" : "Play"}
              disabled={atEnd && !playing}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "▮▮" : "▶"}
            </button>
            <button className={s.ctrl} title="Step forward" disabled={atEnd} onClick={() => seek(state.move + 1)}>
              ▶▏
            </button>
            <button className={s.ctrl} title="Skip to end" disabled={atEnd} onClick={() => seek(state.n_moves)}>
              ⏭
            </button>
            <input
              type="range"
              min={0}
              max={state.n_moves}
              value={state.move}
              onChange={(e) => seek(Number(e.target.value))}
              className={s.slider}
            />
            <span className={s.counter}>
              move {state.move} / {state.n_moves}
            </span>
            {last && (
              <span className={`fade-in ${s.lastMove}`} key={last.id}>
                <span
                  className={s.lastDot}
                  style={{
                    background: last.player != null ? (PLAYER_COLORS[last.player] ?? "#888") : "#888",
                  }}
                />
                <span className={s.lastIcon}>
                  {last.kind === "win" ? "🏆" : actionMeta(last.action_type ?? "").icon}
                </span>
                <span className={s.lastText}>{last.text}</span>
              </span>
            )}
            <span className={s.loadGroup}>
              <LoadButtons small onLoad={loaded} onError={setError} />
              <a
                href="/api/replay/record"
                download="settlrl-game.json"
                title="Save this record to a file"
                className={s.saveLink}
              >
                💾
              </a>
            </span>
            {error && <span className={s.error}>{error}</span>}
          </div>
        </div>
      </div>

      <ChatPanel entries={state.log} title="Log" players={state.board.players} />
    </div>
  );
}
