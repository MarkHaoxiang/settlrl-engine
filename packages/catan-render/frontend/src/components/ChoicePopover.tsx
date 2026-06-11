import type { GameAction } from "../lib/game";
import { DIVIDER, buttonStyle, smallButtonStyle } from "../lib/ui";

// A row in the bottom panel listing concrete actions to pick from (resource
// choices: monopoly, year of plenty, maritime trade).
export default function ChoicePopover({
  actions,
  onPick,
  onClose,
}: {
  actions: GameAction[];
  onPick: (flat: number) => void;
  onClose: () => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, borderTop: `1px solid ${DIVIDER}`, paddingTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 }}>Choose</span>
        <button style={{ ...smallButtonStyle, padding: "2px 10px" }} onClick={onClose}>
          Cancel
        </button>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", maxWidth: 640 }}>
        {actions.map((a) => (
          <button key={a.flat} style={buttonStyle} onClick={() => onPick(a.flat)}>
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}
