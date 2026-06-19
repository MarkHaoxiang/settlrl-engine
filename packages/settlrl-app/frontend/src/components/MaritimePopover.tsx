import type { GameAction } from "../lib/game";
import {
  RESOURCE_LABELS,
  RESOURCE_ORDER,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  maritimeRate,
  type Board,
  type ResourceKind,
} from "../lib/boardData";
import Anchored from "./Anchored";
import CountBadge from "./CountBadge";
import s from "./MaritimePopover.module.css";
import ResourceGlyph from "./ResourceGlyph";

// Anchored at the bank pile of the resource you clicked: pick what to give for
// one of it. Each give option is an icon chip badged with its rate — how many
// you hand over (4:1, or 3/2 through your ports) — so the cost reads at a
// glance instead of as "Trade wood → wheat" text.
export default function MaritimePopover({
  receive,
  actions,
  board,
  player,
  x,
  y,
  disabled,
  onPick,
  onClose,
}: {
  receive: ResourceKind;
  // The legal maritime_trade actions that yield this resource.
  actions: GameAction[];
  board: Board;
  player: number;
  x: number;
  y: number;
  disabled?: boolean;
  onPick: (flat: number) => void;
  onClose: () => void;
}) {
  // One option per giveable resource, in the shared resource order.
  const options = RESOURCE_ORDER.map((give) => actions.find((a) => a.give === give))
    .filter((a): a is GameAction => a !== undefined)
    .map((a) => ({ action: a, give: a.give as ResourceKind }));

  return (
    <Anchored x={x} y={y} onClose={onClose}>
      <span className={s.header}>
        Trade for 1
        <ResourceGlyph kind={receive} px={20} scale={0.95} />
        {RESOURCE_LABELS[receive]}
      </span>
      <div className={s.options}>
        {options.map(({ action, give }) => {
          const rate = maritimeRate(board, player, give);
          const fill = TERRAIN_FILL[give];
          return (
            <button
              key={action.flat}
              title={`Give ${rate} ${RESOURCE_LABELS[give].toLowerCase()} for 1 ${RESOURCE_LABELS[receive].toLowerCase()}`}
              disabled={disabled}
              onClick={() => onPick(action.flat)}
              className={s.chip}
              style={{
                background: fill,
                border: `2px solid ${TERRAIN_STROKE[give]}`,
                cursor: disabled ? "default" : "pointer",
              }}
            >
              <ResourceGlyph kind={give} px={26} scale={1.15} />
              <CountBadge value={`×${rate}`} />
            </button>
          );
        })}
      </div>
    </Anchored>
  );
}
