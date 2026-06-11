import { useState } from "react";
import type { GameAction } from "../lib/game";
import {
  RESOURCE_LABELS,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  playerName,
  type ResourceKind,
} from "../lib/boardData";
import { ACCENT } from "../lib/ui";
import Anchored from "./Anchored";
import TerrainIcon from "./TerrainIcon";

function ResourceChipRow({
  options,
  selected,
  onPick,
}: {
  options: ResourceKind[];
  selected?: ResourceKind | null;
  onPick: (r: ResourceKind) => void;
}) {
  return (
    <span style={{ display: "inline-flex", gap: 4 }}>
      {options.map((r) => (
        <button
          key={r}
          title={RESOURCE_LABELS[r]}
          style={{
            width: 30,
            height: 26,
            borderRadius: 6,
            background: TERRAIN_FILL[r],
            border: `2px solid ${TERRAIN_STROKE[r]}`,
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 0,
            ...(selected === r ? { outline: `2px solid ${ACCENT}`, outlineOffset: 1 } : {}),
          }}
          onClick={() => onPick(r)}
        >
          <svg width={20} height={20} viewBox="-11 -11 22 22">
            <TerrainIcon terrain={r} cx={0} cy={0} scale={0.95} opacity={0.9} />
          </svg>
        </button>
      ))}
    </span>
  );
}

// Anchored at an opponent's hand pile: propose a 1:1 trade by picking the
// card you give, then clicking the card you want (which fires the proposal —
// the partner still has to accept).
export default function TradePopover({
  partner,
  actions,
  x,
  y,
  onPick,
  onClose,
}: {
  partner: number;
  // The legal propose_trade actions towards this partner.
  actions: GameAction[];
  x: number;
  y: number;
  onPick: (flat: number) => void;
  onClose: () => void;
}) {
  const gives = [...new Set(actions.map((a) => a.give as ResourceKind))];
  const [give, setGive] = useState<ResourceKind | null>(gives.length === 1 ? gives[0] : null);
  const receives = give
    ? actions.filter((a) => a.give === give).map((a) => a.receive as ResourceKind)
    : [];

  const label = { fontSize: 11, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 } as const;
  return (
    <Anchored x={x} y={y} onClose={onClose}>
      <span style={{ fontSize: 13, fontWeight: 700 }}>Offer {playerName(partner)} a trade</span>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ ...label, width: 58 }}>You give</span>
        <ResourceChipRow options={gives} selected={give} onPick={setGive} />
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, opacity: give ? 1 : 0.35 }}>
        <span style={{ ...label, width: 58 }}>You get</span>
        {give ? (
          <ResourceChipRow
            options={receives}
            onPick={(r) => {
              const m = actions.find((a) => a.give === give && a.receive === r);
              if (m) onPick(m.flat);
            }}
          />
        ) : (
          <span style={{ fontSize: 11, opacity: 0.7 }}>pick a card to give first</span>
        )}
      </div>
    </Anchored>
  );
}
