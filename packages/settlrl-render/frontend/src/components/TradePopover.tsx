import { useState } from "react";
import type { GameAction, PlayerBelief } from "../lib/game";
import {
  RESOURCE_LABELS,
  RESOURCE_ORDER,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  playerName,
  type Player,
  type ResourceKind,
} from "../lib/boardData";
import { ACCENT, buttonStyle } from "../lib/ui";
import Anchored from "./Anchored";
import CountBadge from "./CountBadge";
import ResourceGlyph from "./ResourceGlyph";

const boundText = (lo: number, hi: number): string => (lo === hi ? `${lo}` : `${lo}–${hi}`);

// One resource in a trade column: the terrain chip with an annotation overlaid
// (your count, or what card counting proves about theirs).
function TradeChip({
  r,
  annotation,
  selected,
  disabled,
  onClick,
}: {
  r: ResourceKind;
  annotation: string;
  selected: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  const fill = TERRAIN_FILL[r];
  return (
    <button
      title={`${RESOURCE_LABELS[r]}${disabled ? " — none to give" : ""}`}
      disabled={disabled}
      style={{
        position: "relative",
        width: 38,
        height: 32,
        borderRadius: 7,
        background: fill,
        border: `2px solid ${TERRAIN_STROKE[r]}`,
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.3 : 1,
        padding: 0,
        ...(selected ? { outline: `2px solid ${ACCENT}`, outlineOffset: 1 } : {}),
      }}
      onClick={onClick}
    >
      <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <ResourceGlyph kind={r} px={24} opacity={0.85} />
      </span>
      <CountBadge value={annotation} />
    </button>
  );
}

function Side({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <span style={{ fontSize: 11, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 }}>{title}</span>
      <div style={{ display: "flex", gap: 4 }}>{children}</div>
    </div>
  );
}

// Anchored at an opponent's hand pile: the two-sided 1:1 offer composer.
// Your column shows your hand counts (cards you can give are selectable);
// theirs shows every resource you may ask for, annotated with the proven
// bounds card counting gives you on their hand. Propose fires the offer —
// the partner still has to accept.
export default function TradePopover({
  partner,
  actions,
  me,
  bounds,
  x,
  y,
  disabled,
  onPick,
  onClose,
}: {
  partner: number;
  // The legal propose_trade actions towards this partner.
  actions: GameAction[];
  me: Player;
  // Card counting on the partner's hand, when available.
  bounds?: PlayerBelief;
  x: number;
  y: number;
  disabled?: boolean;
  onPick: (flat: number) => void;
  onClose: () => void;
}) {
  const [give, setGive] = useState<ResourceKind | null>(null);
  const [receive, setReceive] = useState<ResourceKind | null>(null);
  const givable = new Set(actions.map((a) => a.give as ResourceKind));
  const match =
    give && receive ? actions.find((a) => a.give === give && a.receive === receive) : undefined;

  return (
    <Anchored x={x} y={y} onClose={onClose}>
      <span style={{ fontSize: 13, fontWeight: 700 }}>Trade with {playerName(partner)} — one card each way</span>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
        <Side title="You give">
          {RESOURCE_ORDER.map((r) => (
            <TradeChip
              key={r}
              r={r}
              annotation={`${me.resources?.[r] ?? 0}`}
              selected={give === r}
              disabled={!givable.has(r)}
              onClick={() => setGive(give === r ? null : r)}
            />
          ))}
        </Side>
        <span style={{ fontSize: 20, opacity: 0.7, paddingBottom: 4 }}>⇄</span>
        <Side title={`${playerName(partner)} gives`}>
          {RESOURCE_ORDER.map((r) => (
            <TradeChip
              key={r}
              r={r}
              annotation={bounds ? boundText(bounds.res_lo[r], bounds.res_hi[r]) : "?"}
              selected={receive === r}
              onClick={() => setReceive(receive === r ? null : r)}
            />
          ))}
        </Side>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 12, opacity: 0.75, flex: 1 }}>
          {give && receive
            ? `1 ${RESOURCE_LABELS[give].toLowerCase()} for 1 ${RESOURCE_LABELS[receive].toLowerCase()}`
            : "pick a card from each side"}
        </span>
        <button
          disabled={disabled || !match}
          style={{
            ...buttonStyle,
            fontSize: 13,
            padding: "6px 14px",
            ...(match ? { borderColor: ACCENT } : { opacity: 0.5, cursor: "default" }),
          }}
          onClick={() => match && onPick(match.flat)}
        >
          🤝 Propose
        </button>
      </div>
    </Anchored>
  );
}
