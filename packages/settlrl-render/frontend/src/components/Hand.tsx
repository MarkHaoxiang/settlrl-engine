import {
  DEV_CARD_BACK,
  PLAYER_COLORS,
  PLAYER_STROKES,
  RESOURCE_LABELS,
  RESOURCE_ORDER,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  playerName,
  type DevCardKind,
  type Player,
  type ResourceKind,
} from "../lib/boardData";
import { ACCENT, ACCENT_GLOW, DIVIDER } from "../lib/ui";
import TerrainIcon from "./TerrainIcon";

const DEV_CARDS: { key: DevCardKind; label: string; icon: string }[] = [
  { key: "knight", label: "Knight", icon: "⚔️" },
  { key: "road_building", label: "Road building", icon: "🚧" },
  { key: "year_of_plenty", label: "Year of plenty", icon: "🎁" },
  { key: "monopoly", label: "Monopoly", icon: "🎩" },
  { key: "victory_point", label: "Victory point", icon: "⭐" },
];

// The play action behind each dev-card hand chip (victory points are never
// played, so they have no entry).
export const DEV_PLAY_TYPE: Partial<Record<DevCardKind, string>> = {
  knight: "play_knight",
  road_building: "play_road_building",
  year_of_plenty: "play_year_of_plenty",
  monopoly: "play_monopoly",
};

// A hand chip: the count over a faded background icon (the card's name is the
// hover tooltip). Clickable chips glow; `selected` marks an armed card (the
// knight while choosing its robber tile).
function Chip({
  count,
  label,
  icon,
  fill,
  stroke,
  onClick,
  selected,
}: {
  count: number;
  label: string;
  icon: React.ReactNode;
  fill: string;
  stroke: string;
  onClick?: () => void;
  selected?: boolean;
}) {
  return (
    <div
      title={label}
      onClick={onClick}
      style={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 40,
        height: 34,
        borderRadius: 8,
        background: fill,
        border: `2px solid ${stroke}`,
        opacity: count === 0 ? 0.4 : 1,
        ...(onClick ? { cursor: "pointer", boxShadow: ACCENT_GLOW } : {}),
        ...(selected ? { outline: `2px solid ${ACCENT}`, outlineOffset: 1 } : {}),
      }}
    >
      <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {icon}
      </span>
      <span
        style={{
          position: "relative",
          fontSize: 16,
          fontWeight: 800,
          lineHeight: 1,
          color: "#1a1a1a",
          // A halo in the chip colour keeps the digit legible over the icon.
          textShadow: `0 0 4px ${fill}, 0 0 4px ${fill}, 0 0 3px ${fill}`,
        }}
      >
        {count}
      </span>
    </div>
  );
}

// The acting human's hand: resources + dev cards by type, on a single row.
// Chips double as controls: dev cards in `playableDev` play on click, and
// resources in `discardable` discard one on click.
export default function Hand({
  player,
  you,
  discardable,
  onDiscard,
  playableDev,
  armedDev,
  onDev,
}: {
  player: Player;
  you: boolean;
  discardable?: Set<ResourceKind>;
  onDiscard?: (r: ResourceKind) => void;
  playableDev?: Set<DevCardKind>;
  armedDev?: DevCardKind | null;
  onDev?: (k: DevCardKind) => void;
}) {
  const color = PLAYER_COLORS[player.player] ?? "#888";
  const stroke = PLAYER_STROKES[player.player] ?? "#444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginRight: 8 }}>
        <span style={{ width: 14, height: 14, borderRadius: "50%", background: color, border: `2px solid ${stroke}` }} />
        <span style={{ fontWeight: 700, fontSize: 13 }}>
          {playerName(player.player)}
          {you ? " (you)" : ""}
        </span>
      </div>
      {RESOURCE_ORDER.map((r) => {
        const canDiscard = discardable?.has(r) ?? false;
        return (
          <Chip
            key={r}
            count={player.resources?.[r] ?? 0}
            label={canDiscard ? `${RESOURCE_LABELS[r]} — click to discard one` : RESOURCE_LABELS[r]}
            // The board tiles' motif, so chips match the terrain they come from.
            icon={
              <svg width={28} height={28} viewBox="-11 -11 22 22">
                <TerrainIcon terrain={r} cx={0} cy={0} scale={1.1} opacity={0.9} />
              </svg>
            }
            fill={TERRAIN_FILL[r]}
            stroke={TERRAIN_STROKE[r]}
            onClick={canDiscard ? () => onDiscard?.(r) : undefined}
          />
        );
      })}
      <span style={{ width: 1, alignSelf: "stretch", background: DIVIDER, margin: "0 8px" }} />
      {DEV_CARDS.map((d) => {
        const canPlay = playableDev?.has(d.key) ?? false;
        return (
          <Chip
            key={d.key}
            count={player.devCardTypes?.[d.key] ?? 0}
            label={canPlay ? `${d.label} — click to play` : d.label}
            icon={<span style={{ fontSize: 17, opacity: 0.8 }}>{d.icon}</span>}
            fill={DEV_CARD_BACK.fill}
            stroke={DEV_CARD_BACK.stroke}
            onClick={canPlay ? () => onDev?.(d.key) : undefined}
            selected={armedDev === d.key}
          />
        );
      })}
    </div>
  );
}
