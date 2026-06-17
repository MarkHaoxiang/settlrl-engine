import { actionMeta, confirmLabel } from "../lib/actionMeta";
import type { GameAction } from "../lib/game";
import { TERRAIN_FILL, TERRAIN_STROKE, type ResourceKind } from "../lib/boardData";
import { buttonStyle } from "../lib/ui";
import Anchored from "./Anchored";
import ResourceGlyph from "./ResourceGlyph";

// A build price as a row of mini resource chips.
function CostRow({ cost }: { cost: ResourceKind[] }) {
  return (
    <span style={{ display: "inline-flex", gap: 2, marginLeft: 4 }}>
      {cost.map((r, i) => (
        <span
          key={i}
          title={r}
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            background: TERRAIN_FILL[r],
            border: `1px solid ${TERRAIN_STROKE[r]}`,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <ResourceGlyph kind={r} px={12} scale={0.9} opacity={0.5} />
        </span>
      ))}
    </span>
  );
}

interface Props {
  // Anchor point in the board container's coordinate space (BoardTargetPoint).
  x: number;
  y: number;
  // The concrete moves at the clicked target, one confirm button each.
  actions: GameAction[];
  // Per-action price to display, if any (e.g. hidden for free roads).
  costFor: (a: GameAction) => ResourceKind[] | undefined;
  disabled?: boolean;
  onPick: (flat: number) => void;
  onClose: () => void;
}

// The chooser anchored to a clicked board element, confirming a move (with
// its build cost) or picking between variants (robber victims, bank rates).
export default function BoardPopover({ x, y, actions, costFor, disabled, onPick, onClose }: Props) {
  return (
    <Anchored x={x} y={y} onClose={onClose}>
      {actions.map((a) => {
        const cost = costFor(a);
        return (
          <button
            key={a.flat}
            disabled={disabled}
            style={{ ...buttonStyle, display: "flex", alignItems: "center", gap: 8, whiteSpace: "nowrap" }}
            onClick={() => onPick(a.flat)}
          >
            <span style={{ fontSize: 16 }}>{actionMeta(a.type).icon}</span>
            <span>{confirmLabel(a)}</span>
            {cost && <CostRow cost={cost} />}
          </button>
        );
      })}
    </Anchored>
  );
}
