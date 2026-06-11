import { actionMeta, confirmLabel } from "../lib/actionMeta";
import type { GameAction } from "../lib/game";
import { TERRAIN_FILL, TERRAIN_STROKE, type ResourceKind } from "../lib/boardData";
import { buttonStyle, panelStyle } from "../lib/ui";
import TerrainIcon from "./TerrainIcon";

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
          <svg width={12} height={12} viewBox="-11 -11 22 22">
            <TerrainIcon terrain={r} cx={0} cy={0} scale={0.9} />
          </svg>
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
// its build cost) or picking between variants (robber victims). The full-size
// backdrop closes it on any outside press, so a stray click can't fire a
// second action; it also blocks pan/zoom while open (a wheel just closes it).
export default function BoardPopover({ x, y, actions, costFor, disabled, onPick, onClose }: Props) {
  // Near the top edge, open downward instead of clipping off-screen.
  const below = y < 150;
  return (
    <div
      style={{ position: "absolute", inset: 0, zIndex: 20 }}
      onPointerDown={onClose}
      onWheel={onClose}
    >
      <div
        onPointerDown={(e) => e.stopPropagation()}
        style={{
          ...panelStyle,
          position: "absolute",
          left: x,
          top: y,
          transform: below ? "translate(-50%, 14px)" : "translate(-50%, calc(-100% - 14px))",
          display: "flex",
          flexDirection: "column",
          gap: 6,
          padding: 10,
          boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
        }}
      >
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
      </div>
    </div>
  );
}
