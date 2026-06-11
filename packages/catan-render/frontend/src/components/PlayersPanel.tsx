import { useState } from "react";
import {
  PLAYER_COLORS,
  PLAYER_STROKES,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  playerName,
  type Player,
  type ResourceKind,
} from "../lib/boardData";
import type { Belief } from "../lib/game";
import { buttonStyle, selectedStyle } from "../lib/ui";

const RES_ORDER: ResourceKind[] = ["wood", "brick", "sheep", "wheat", "ore"];

// One proven bound, "n" when exact or "lo–hi" when a steal blurred it.
const boundText = (lo: number, hi: number): string => (lo === hi ? `${lo}` : `${lo}–${hi}`);

// The seat list in playing order: every player's public counts on one row
// (⭐ victory points, 🎴 resource cards, 🃏 dev cards), the acting seat
// tinted. 🔍 expands an opponent's card-counting view — the proven
// per-resource bounds the observing human could have tracked themself.
export default function PlayersPanel({
  players,
  acting,
  you,
  belief,
}: {
  players: Player[];
  acting?: number;
  // The hand-panel seat, marked "(you)".
  you?: number;
  belief?: Belief | null;
}) {
  const [inspected, setInspected] = useState<number | null>(null);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3, padding: "12px 10px 8px" }}>
      <span
        style={{
          fontSize: 11,
          opacity: 0.6,
          textTransform: "uppercase",
          letterSpacing: 1,
          padding: "0 4px 4px",
        }}
      >
        Players
      </span>
      {players.map((p) => {
        const b = belief?.players.find((x) => x.player === p.player);
        const open = inspected === p.player && b != null;
        return (
          <div
            key={p.player}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 5,
              borderRadius: 8,
              padding: "5px 8px",
              ...(p.player === acting ? { background: "var(--selected-bg)" } : {}),
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <span
                style={{
                  width: 11,
                  height: 11,
                  borderRadius: "50%",
                  flexShrink: 0,
                  background: PLAYER_COLORS[p.player] ?? "#888",
                  border: `2px solid ${PLAYER_STROKES[p.player] ?? "#444"}`,
                }}
              />
              <span style={{ fontWeight: 700, whiteSpace: "nowrap" }}>
                {playerName(p.player)}
                {p.player === you ? " (you)" : ""}
              </span>
              <span style={{ marginLeft: "auto", display: "flex", gap: 7, opacity: 0.95, whiteSpace: "nowrap" }}>
                <span title="victory points">⭐{p.victoryPoints}</span>
                <span title="resource cards">🎴{p.resourceCards}</span>
                <span title="development cards">🃏{p.devCards}</span>
              </span>
              {b && (
                <button
                  title="Inspect: proven hand bounds (card counting)"
                  style={{
                    ...buttonStyle,
                    padding: "0 5px",
                    fontSize: 11,
                    lineHeight: 1.6,
                    ...(open ? selectedStyle : {}),
                  }}
                  onClick={() => setInspected(open ? null : p.player)}
                >
                  🔍
                </button>
              )}
            </div>
            {open && b && (
              <div style={{ display: "flex", gap: 3, paddingLeft: 19 }} className="fade-in">
                {RES_ORDER.map((r) => (
                  <span
                    key={r}
                    title={`${r}: ${boundText(b.res_lo[r], b.res_hi[r])}`}
                    style={{
                      minWidth: 26,
                      textAlign: "center",
                      borderRadius: 4,
                      padding: "1px 3px",
                      fontSize: 10,
                      fontWeight: 700,
                      background: TERRAIN_FILL[r],
                      border: `1px solid ${TERRAIN_STROKE[r]}`,
                      color: "#1a1a1a",
                    }}
                  >
                    {boundText(b.res_lo[r], b.res_hi[r])}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
