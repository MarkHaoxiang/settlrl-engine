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
import { HIGHLIGHT, buttonStyle, panelStyle } from "../lib/ui";

const PLAY_INTERVAL_MS = 600;

const btnStyle: React.CSSProperties = {
  ...buttonStyle,
  width: 36,
  height: 32,
  padding: 0,
  fontSize: 14,
  lineHeight: 1,
};

// The pair of ways to load a record: a saved .json file, or the live game as
// played so far. Rendered both on the empty state and (small) in the bar.
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
  const style = small ? { ...btnStyle, width: undefined, padding: "0 10px" } : buttonStyle;

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
      <button style={style} title="Open a saved record file" onClick={() => fileRef.current?.click()}>
        {small ? "📂" : "📂 Open record file…"}
      </button>
      <button
        style={style}
        title="Replay the live game as played so far"
        onClick={() => loadReplayFromGame().then(onLoad, (e: unknown) => onError(String(e)))}
      >
        {small ? "🎮" : "🎮 Use current game"}
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
      <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
        <TopBar mode="Replay" />
        <div
          style={{
            ...panelStyle,
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            display: "flex",
            flexDirection: "column",
            gap: 14,
            padding: "26px 30px",
            alignItems: "center",
            maxWidth: 380,
          }}
        >
          <span style={{ fontWeight: 700, fontSize: 16 }}>Replay a game</span>
          <span style={{ fontSize: 13, opacity: 0.75, textAlign: "center" }}>
            Open a saved game record, or replay the game currently in Play from its first move.
          </span>
          <div style={{ display: "flex", gap: 10 }}>
            <LoadButtons onLoad={loaded} onError={setError} />
          </div>
          {error && <span style={{ color: "var(--error)", fontSize: 12 }}>{error}</span>}
        </div>
      </div>
    );
  }

  const atEnd = state.move >= state.n_moves;
  const last = state.move > 0 ? state.log[state.log.length - 1] : undefined;

  return (
    <div style={{ display: "flex", width: "100vw", height: "100vh", overflow: "hidden" }}>
      <div style={{ position: "relative", flex: 1, overflow: "hidden" }}>
        <BoardView board={state.board} />
        <TopBar mode="Replay" />

        {atEnd && state.winner != null && (
          <div
            style={{
              ...panelStyle,
              position: "absolute",
              top: 70,
              left: "50%",
              transform: "translateX(-50%)",
              padding: "10px 20px",
              color: HIGHLIGHT,
              fontWeight: 700,
              zIndex: 10,
            }}
          >
            {playerName(state.winner)} wins
          </div>
        )}

        {/* Bottom centre: playback controls */}
        <div
          style={{
            position: "absolute",
            bottom: 16,
            left: 0,
            right: 0,
            display: "flex",
            justifyContent: "center",
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              ...panelStyle,
              pointerEvents: "auto",
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 16px",
              borderRadius: 16,
              boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
              maxWidth: "94%",
              flexWrap: "wrap",
              justifyContent: "center",
            }}
          >
            <button style={btnStyle} title="Back to start" disabled={state.move === 0} onClick={() => seek(0)}>
              ⏮
            </button>
            <button style={btnStyle} title="Step back" disabled={state.move === 0} onClick={() => seek(state.move - 1)}>
              ◀
            </button>
            <button
              style={btnStyle}
              title={playing ? "Pause" : "Play"}
              disabled={atEnd && !playing}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "▮▮" : "▶"}
            </button>
            <button style={btnStyle} title="Step forward" disabled={atEnd} onClick={() => seek(state.move + 1)}>
              ▶▏
            </button>
            <button style={btnStyle} title="Skip to end" disabled={atEnd} onClick={() => seek(state.n_moves)}>
              ⏭
            </button>
            <input
              type="range"
              min={0}
              max={state.n_moves}
              value={state.move}
              onChange={(e) => seek(Number(e.target.value))}
              style={{ width: 220 }}
            />
            <span style={{ fontSize: 12, opacity: 0.7, minWidth: 86, textAlign: "center" }}>
              move {state.move} / {state.n_moves}
            </span>
            {last && (
              <span
                className="fade-in"
                key={last.id}
                style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13 }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: last.player != null ? (PLAYER_COLORS[last.player] ?? "#888") : "#888",
                  }}
                />
                <span style={{ fontSize: 16 }}>
                  {last.kind === "win" ? "🏆" : actionMeta(last.action_type ?? "").icon}
                </span>
                <span style={{ opacity: 0.8 }}>{last.text}</span>
              </span>
            )}
            <span style={{ display: "inline-flex", gap: 6, marginLeft: 8 }}>
              <LoadButtons small onLoad={loaded} onError={setError} />
              <a
                href="/api/replay/record"
                download="catan-game.json"
                title="Save this record to a file"
                style={{ ...btnStyle, width: undefined, padding: "0 10px", display: "inline-flex", alignItems: "center", textDecoration: "none" }}
              >
                💾
              </a>
            </span>
            {error && <span style={{ color: "var(--error)", fontSize: 12 }}>{error}</span>}
          </div>
        </div>
      </div>

      <ChatPanel entries={state.log} title="Log" />
    </div>
  );
}
