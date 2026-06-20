import { useEffect, useRef, useState } from "react";
import { actionMeta } from "../lib/actionMeta";
import { PLAYER_COLORS, playerName, type Player } from "../lib/boardData";
import type { Belief, LogEntry } from "../lib/game";
import s from "./ChatPanel.module.css";
import PlayersPanel from "./PlayersPanel";

interface Props {
  entries: LogEntry[];
  // Omit to render a read-only log (no input row), e.g. for replays.
  onSend?: (text: string) => void;
  title?: string;
  // When set, the column opens with the seat list (playing order) above the
  // log; `acting` / `you` / `belief` flow through to it.
  players?: Player[];
  acting?: number;
  you?: number;
  belief?: Belief | null;
  // Per-seat owner labels (account name / "Guest" / bot kind) for the seat list.
  identities?: (string | null)[];
}

// Right-hand column: the seat list on top (when given), then the chat & game
// log rendering the server-side log — every move lands here as an event line,
// and humans can type messages.
export default function ChatPanel({ entries, onSend, title = "Chat", players, acting, you, belief, identities }: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the newest line in view as the log grows.
  useEffect(() => {
    endRef.current?.scrollIntoView();
  }, [entries]);

  const send = () => {
    const text = draft.trim();
    if (text && onSend) onSend(text);
    setDraft("");
  };

  return (
    <div className={s.panel}>
      {players && (
        <>
          <PlayersPanel players={players} acting={acting} you={you} belief={belief} identities={identities} />
          <div className={s.divider} />
        </>
      )}
      <span className={s.title}>{title}</span>
      <div className={s.log}>
        {entries.map((m) => (
          <div
            key={m.id}
            className={`fade-in ${m.kind === "move" ? s.entryMove : s.entry}`}
          >
            <span
              className={s.dot}
              style={{
                background: m.player != null ? (PLAYER_COLORS[m.player] ?? "#888") : "#888",
              }}
            />
            {m.kind === "win" && <span>🏆</span>}
            {m.action_type && <span>{actionMeta(m.action_type).icon}</span>}
            {m.kind !== "move" && (
              <span className={s.author}>
                {m.player != null ? playerName(m.player) : "Spectator"}
              </span>
            )}
            <span className={s.text}>{m.text}</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      {onSend && (
        <div className={s.inputRow}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Say something…"
            className={s.input}
          />
        </div>
      )}
    </div>
  );
}
