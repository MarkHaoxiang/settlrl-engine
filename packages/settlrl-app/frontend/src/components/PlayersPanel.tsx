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
import s from "./PlayersPanel.module.css";

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
      className={s.glyph}
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
    <span title={label} className={s.stat}>
      {glyph}
      <span className={s.statValue}>{value}</span>
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
  identities,
}: {
  players: Player[];
  acting?: number;
  // The hand-panel seat, marked "(you)".
  you?: number;
  belief?: Belief | null;
  // Who holds each seat: an account name or "Guest" for a human, the bot kind
  // for a bot. Indexed by seat; absent renders just the colour name.
  identities?: (string | null)[];
}) {
  const [inspected, setInspected] = useState<number | null>(null);
  return (
    <div className={s.panel}>
      <span className={s.label}>Players</span>
      {players.map((p) => {
        const b = belief?.players.find((x) => x.player === p.player);
        const open = inspected === p.player && b != null;
        return (
          <div key={p.player} className={p.player === acting ? s.rowActive : s.row}>
            <div className={s.line}>
              <span
                className={s.dot}
                style={{
                  background: PLAYER_COLORS[p.player] ?? "#888",
                  border: `1.5px solid ${PLAYER_STROKES[p.player] ?? "#444"}`,
                }}
              />
              <span className={s.name}>
                {playerName(p.player)}
                {p.player === you ? " (you)" : ""}
              </span>
              {identities?.[p.player] ? (
                <span className={s.identity} title={identities[p.player] ?? undefined}>
                  {identities[p.player]}
                </span>
              ) : null}
              <span className={s.stats}>
                <Stat glyph={<VpGlyph />} value={p.victoryPoints} label="victory points" />
                <Stat glyph={<CardsGlyph />} value={p.resourceCards} label="resource cards" />
                <Stat glyph={<DevGlyph />} value={p.devCards} label="development cards" />
              </span>
              {b ? (
                <button
                  title="Inspect: proven hand bounds (card counting)"
                  className={open ? s.inspectOpen : s.inspect}
                  onClick={() => setInspected(open ? null : p.player)}
                >
                  <MagnifierGlyph />
                </button>
              ) : (
                // Keep the stat columns aligned on rows without an inspect.
                <span className={s.spacer} />
              )}
            </div>
            {open && b && (
              <div className={`fade-in ${s.bounds}`}>
                {RESOURCE_ORDER.map((r) => (
                  <span
                    key={r}
                    title={`${r}: ${boundText(b.res_lo[r], b.res_hi[r])}`}
                    className={s.boundChip}
                    style={{
                      background: TERRAIN_FILL[r],
                      border: `1px solid ${TERRAIN_STROKE[r]}`,
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
