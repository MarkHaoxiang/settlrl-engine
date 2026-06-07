import { useEffect, useRef, useState } from "react";
import { actionMeta } from "../lib/actionMeta";
import { PLAYER_COLORS, playerName } from "../lib/boardData";
import type { LogEntry } from "../lib/game";
import { buttonStyle, panelStyle } from "../lib/ui";

interface Props {
  entries: LogEntry[];
  onSend: (text: string) => void;
}

// Right-hand chat & game log column, rendering the server-side log: every
// move lands here as an event line, and humans can type messages.
export default function ChatPanel({ entries, onSend }: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the newest line in view as the log grows.
  useEffect(() => {
    endRef.current?.scrollIntoView();
  }, [entries]);

  const send = () => {
    const text = draft.trim();
    if (text) onSend(text);
    setDraft("");
  };

  return (
    <div
      style={{
        ...panelStyle,
        width: 260,
        margin: 12,
        marginLeft: 0,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <span
        style={{
          fontSize: 11,
          opacity: 0.6,
          textTransform: "uppercase",
          letterSpacing: 1,
          padding: "12px 14px 8px",
        }}
      >
        Chat
      </span>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 6,
          padding: "2px 14px",
        }}
      >
        {entries.map((m) => (
          <div
            key={m.id}
            className="fade-in"
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: 6,
              fontSize: 12,
              opacity: m.kind === "move" ? 0.75 : 1,
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                flexShrink: 0,
                alignSelf: "center",
                background: m.player != null ? (PLAYER_COLORS[m.player] ?? "#888") : "#888",
              }}
            />
            {m.kind === "win" && <span>🏆</span>}
            {m.action_type && <span>{actionMeta(m.action_type).icon}</span>}
            {m.kind !== "move" && (
              <span style={{ fontWeight: 700 }}>
                {m.player != null ? playerName(m.player) : "Spectator"}
              </span>
            )}
            <span style={{ overflowWrap: "anywhere" }}>{m.text}</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div style={{ padding: 10 }}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Say something…"
          style={{ ...buttonStyle, cursor: "text", width: "100%", fontSize: 12, padding: "7px 10px" }}
        />
      </div>
    </div>
  );
}
