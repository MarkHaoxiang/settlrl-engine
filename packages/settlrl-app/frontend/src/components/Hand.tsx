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
import CountBadge from "./CountBadge";
import s from "./Hand.module.css";
import ResourceGlyph from "./ResourceGlyph";

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
      className={s.chip}
      style={{
        background: fill,
        border: `2px solid ${stroke}`,
        opacity: count === 0 ? 0.4 : 1,
        ...(onClick ? { cursor: "pointer", boxShadow: "var(--accent-glow-shadow)" } : {}),
        ...(selected ? { outline: "2px solid var(--accent)", outlineOffset: 1 } : {}),
      }}
    >
      <span className={s.chipIcon}>{icon}</span>
      <CountBadge value={count} />
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
    <div className={s.hand}>
      <div className={s.nameGroup}>
        <span className={s.dot} style={{ background: color, border: `2px solid ${stroke}` }} />
        <span className={s.name}>
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
            icon={<ResourceGlyph kind={r} px={28} scale={1.1} />}
            fill={TERRAIN_FILL[r]}
            stroke={TERRAIN_STROKE[r]}
            onClick={canDiscard ? () => onDiscard?.(r) : undefined}
          />
        );
      })}
      <span className={s.divider} />
      {DEV_CARDS.map((d) => {
        const canPlay = playableDev?.has(d.key) ?? false;
        return (
          <Chip
            key={d.key}
            count={player.devCardTypes?.[d.key] ?? 0}
            label={canPlay ? `${d.label} — click to play` : d.label}
            icon={<span className={s.devIcon}>{d.icon}</span>}
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
