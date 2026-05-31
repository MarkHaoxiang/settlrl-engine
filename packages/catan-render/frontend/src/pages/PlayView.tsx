import GameShell from "../components/GameShell";
import {
  PLAYER_COLORS,
  PLAYER_STROKES,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  type DevCardKind,
  type Player,
  type ResourceKind,
} from "../lib/boardData";

// The player controlled from this view. There's no turn flow yet, so the local
// player is fixed to seat 0 for now.
const LOCAL_PLAYER = 0;
const PLAYER_NAMES = ["Red", "Blue", "White", "Orange"];

// Resources shown left-to-right, with the label under each chip.
const RESOURCES: { key: ResourceKind; label: string }[] = [
  { key: "wood", label: "Wood" },
  { key: "brick", label: "Brick" },
  { key: "sheep", label: "Sheep" },
  { key: "wheat", label: "Wheat" },
  { key: "ore", label: "Ore" },
];

// Dev cards use a shared parchment-purple chip; labels kept short to fit.
const DEV_CARDS: { key: DevCardKind; label: string }[] = [
  { key: "knight", label: "Knight" },
  { key: "road_building", label: "Roads" },
  { key: "year_of_plenty", label: "Plenty" },
  { key: "monopoly", label: "Mono" },
  { key: "victory_point", label: "VP" },
];

const DEV_FILL = "#5B4B8A";
const DEV_STROKE = "#3C3160";

const barStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
  padding: "14px 18px",
  borderRadius: 16,
  background: "rgba(12, 28, 46, 0.9)",
  border: "1px solid rgba(255,255,255,0.15)",
  color: "#F2EFE6",
  fontFamily: "Georgia, serif",
  backdropFilter: "blur(2px)",
  userSelect: "none",
  boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
};

const actionStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.2)",
  color: "#F2EFE6",
  borderRadius: 8,
  padding: "9px 16px",
  fontSize: 14,
  fontFamily: "Georgia, serif",
  cursor: "pointer",
};

// A single hand chip: large count over a small label, on a coloured swatch.
// Dimmed when the player holds none of that card.
function Chip({ count, label, fill, stroke }: { count: number; label: string; fill: string; stroke: string }) {
  const empty = count === 0;
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minWidth: 46,
        padding: "6px 8px",
        borderRadius: 10,
        background: fill,
        border: `2px solid ${stroke}`,
        opacity: empty ? 0.4 : 1,
      }}
    >
      <span style={{ fontSize: 20, fontWeight: 700, lineHeight: 1, color: "#1a1a1a" }}>{count}</span>
      <span style={{ fontSize: 10, marginTop: 2, color: "#1a1a1a", opacity: 0.8 }}>{label}</span>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 }}>{title}</span>
      <div style={{ display: "flex", gap: 6 }}>{children}</div>
    </div>
  );
}

// The big play control bar: the local player's hand (resources + dev cards)
// above the turn action buttons. Action buttons are presentation stubs until
// the engine action endpoint is wired up.
function PlayControls({ player }: { player: Player }) {
  const actions = ["Build", "Buy dev card", "Trade", "Roll dice", "End turn"];
  const color = PLAYER_COLORS[player.player] ?? "#888";
  const stroke = PLAYER_STROKES[player.player] ?? "#444";
  const name = PLAYER_NAMES[player.player] ?? `Player ${player.player + 1}`;

  return (
    <div style={barStyle}>
      {/* Hand: who you are + resources + dev cards */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: 20, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, paddingBottom: 4 }}>
          <span
            style={{ width: 16, height: 16, borderRadius: "50%", background: color, border: `2px solid ${stroke}` }}
          />
          <span style={{ fontWeight: 700, fontSize: 14 }}>{name} (you)</span>
        </div>

        <Group title="Resources">
          {RESOURCES.map((r) => (
            <Chip
              key={r.key}
              count={player.resources[r.key]}
              label={r.label}
              fill={TERRAIN_FILL[r.key]}
              stroke={TERRAIN_STROKE[r.key]}
            />
          ))}
        </Group>

        <Group title="Dev cards">
          {DEV_CARDS.map((d) => (
            <Chip key={d.key} count={player.devCardTypes[d.key]} label={d.label} fill={DEV_FILL} stroke={DEV_STROKE} />
          ))}
        </Group>
      </div>

      {/* Turn actions */}
      <div style={{ display: "flex", gap: 10, justifyContent: "center", borderTop: "1px solid rgba(255,255,255,0.12)", paddingTop: 12 }}>
        {actions.map((a) => (
          <button key={a} style={actionStyle} disabled>
            {a}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function PlayView() {
  return <GameShell mode="Play" controls={(board) => <PlayControls player={board.players[LOCAL_PLAYER]} />} />;
}
