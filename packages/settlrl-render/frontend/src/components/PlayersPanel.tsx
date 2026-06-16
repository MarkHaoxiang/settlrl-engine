import { useState } from "react";
import {
  PLAYER_COLORS,
  PLAYER_STROKES,
  RESOURCE_ORDER,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  playerName,
  type Player,
} from "../lib/boardData";
import type { Belief } from "../lib/game";

// One proven bound, "n" when exact or "lo–hi" when a steal blurred it.
const boundText = (lo: number, hi: number): string => (lo === hi ? `${lo}` : `${lo}–${hi}`);

// Tiny monochrome outline glyphs (currentColor) so the rows stay quiet.
function Glyph({ children }: { children: React.ReactNode }) {
  return (
    <svg
      width={12}
      height={12}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ opacity: 0.55, flexShrink: 0 }}
    >
      {children}
    </svg>
  );
}

// A trophy: victory points.
const VpGlyph = () => (
  <Glyph>
    <path d="M7 4h10v4.5a5 5 0 0 1-10 0z" />
    <path d="M7 5H4a3.2 3.2 0 0 0 3.2 4.2" />
    <path d="M17 5h3a3.2 3.2 0 0 1-3.2 4.2" />
    <path d="M12 13.5V17" />
    <path d="M8 20.5h8" />
  </Glyph>
);

// Two fanned cards: the resource hand.
const CardsGlyph = () => (
  <Glyph>
    <rect x="8.5" y="4.5" width="11" height="15" rx="2" />
    <path d="M5 8v10.5a2.5 2.5 0 0 0 2.5 2.5H15" />
  </Glyph>
);

// One card with a diamond pip: development cards.
const DevGlyph = () => (
  <Glyph>
    <rect x="6.5" y="4" width="11" height="16" rx="2" />
    <path d="M12 9l2 3-2 3-2-3z" />
  </Glyph>
);

const MagnifierGlyph = () => (
  <Glyph>
    <circle cx="10.5" cy="10.5" r="6.5" />
    <path d="M15.5 15.5L21 21" />
  </Glyph>
);

function Stat({ glyph, value, label }: { glyph: React.ReactNode; value: number; label: string }) {
  return (
    <span title={label} style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
      {glyph}
      <span style={{ minWidth: 10 }}>{value}</span>
    </span>
  );
}

// The seat list in playing order: every player's public counts on one row
// (victory points, resource cards, dev cards), the acting seat tinted. The
// magnifier expands an opponent's card-counting view — the proven
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
    <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "12px 10px 8px" }}>
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
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  flexShrink: 0,
                  background: PLAYER_COLORS[p.player] ?? "#888",
                  border: `1.5px solid ${PLAYER_STROKES[p.player] ?? "#444"}`,
                }}
              />
              <span style={{ fontWeight: 700, whiteSpace: "nowrap" }}>
                {playerName(p.player)}
                {p.player === you ? " (you)" : ""}
              </span>
              <span style={{ marginLeft: "auto", display: "flex", gap: 9, whiteSpace: "nowrap" }}>
                <Stat glyph={<VpGlyph />} value={p.victoryPoints} label="victory points" />
                <Stat glyph={<CardsGlyph />} value={p.resourceCards} label="resource cards" />
                <Stat glyph={<DevGlyph />} value={p.devCards} label="development cards" />
              </span>
              {b ? (
                <button
                  title="Inspect: proven hand bounds (card counting)"
                  style={{
                    background: "none",
                    border: "none",
                    padding: "0 2px",
                    cursor: "pointer",
                    display: "inline-flex",
                    color: open ? "var(--accent)" : "inherit",
                    opacity: open ? 1 : 0.7,
                  }}
                  onClick={() => setInspected(open ? null : p.player)}
                >
                  <MagnifierGlyph />
                </button>
              ) : (
                // Keep the stat columns aligned on rows without an inspect.
                <span style={{ width: 16 }} />
              )}
            </div>
            {open && b && (
              <div style={{ display: "flex", gap: 3, paddingLeft: 18 }} className="fade-in">
                {RESOURCE_ORDER.map((r) => (
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
                      opacity: 0.9,
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
