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
import Anchored from "./Anchored";
import CountBadge from "./CountBadge";
import ResourceGlyph from "./ResourceGlyph";
import s from "./TradePopover.module.css";

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
      className={s.chip}
      style={{
        background: fill,
        border: `2px solid ${TERRAIN_STROKE[r]}`,
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.3 : 1,
        ...(selected ? { outline: "2px solid var(--accent)", outlineOffset: 1 } : {}),
      }}
      onClick={onClick}
    >
      <span className={s.chipIcon}>
        <ResourceGlyph kind={r} px={24} opacity={0.85} />
      </span>
      <CountBadge value={annotation} />
    </button>
  );
}

function Side({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className={s.side}>
      <span className={s.sideTitle}>{title}</span>
      <div className={s.sideChips}>{children}</div>
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
      <span className={s.header}>Trade with {playerName(partner)} — one card each way</span>
      <div className={s.sides}>
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
        <span className={s.arrow}>⇄</span>
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
      <div className={s.footer}>
        <span className={s.summary}>
          {give && receive
            ? `1 ${RESOURCE_LABELS[give].toLowerCase()} for 1 ${RESOURCE_LABELS[receive].toLowerCase()}`
            : "pick a card from each side"}
        </span>
        <button
          disabled={disabled || !match}
          className={match ? s.proposeReady : s.proposeDisabled}
          onClick={() => match && onPick(match.flat)}
        >
          🤝 Propose
        </button>
      </div>
    </Anchored>
  );
}
