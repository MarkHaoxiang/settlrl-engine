import { PLAYER_COLORS, PLAYER_STROKES, playerName, type Player } from "../lib/boardData";
import { panelStyle } from "../lib/ui";

type Corner = "top-left" | "top-right" | "bottom-left" | "bottom-right";

interface Props {
  player: Player;
  corner: Corner;
}

// Anchor each panel to its corner of the viewport.
const CORNER_STYLE: Record<Corner, React.CSSProperties> = {
  "top-left": { top: 16, left: 16 },
  "top-right": { top: 16, right: 16 },
  "bottom-left": { bottom: 16, left: 16 },
  "bottom-right": { bottom: 16, right: 16 },
};

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: 38 }}>
      <span style={{ fontSize: 18, fontWeight: 700, lineHeight: 1 }}>{value}</span>
      <span style={{ fontSize: 10, opacity: 0.75, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </span>
    </div>
  );
}

export default function PlayerPanel({ player, corner }: Props) {
  const color = PLAYER_COLORS[player.player] ?? "#888";
  const stroke = PLAYER_STROKES[player.player] ?? "#444";

  return (
    <div
      style={{
        ...panelStyle,
        position: "absolute",
        ...CORNER_STYLE[corner],
        display: "flex",
        flexDirection: "column",
        gap: 8,
        padding: "10px 14px",
        border: `2px solid ${color}`,
        boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: color,
            border: `2px solid ${stroke}`,
          }}
        />
        <span style={{ fontWeight: 700, fontSize: 14 }}>{playerName(player.player)}</span>
      </div>
      <div style={{ display: "flex", gap: 12 }}>
        <Stat label="cards" value={player.resourceCards} />
        <Stat label="dev" value={player.devCards} />
        <Stat label="vp" value={player.victoryPoints} />
      </div>
    </div>
  );
}
