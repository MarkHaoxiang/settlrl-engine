import { PLAYER_COLORS, PLAYER_STROKES } from "../lib/boardData";

interface Props {
  cx: number;
  cy: number;
  size: number;
  player: number;
  kind: "settlement" | "city";
}

// A simple house silhouette (square body + triangular roof), centred at (cx, cy)
// and scaled by `size`. Cities are drawn larger to read as upgraded settlements.
// Exported for BoardView's ghost previews of legal placements.
export function housePath(cx: number, cy: number, size: number): string {
  const w = size;
  const h = size;
  const left = cx - w / 2;
  const right = cx + w / 2;
  const bottom = cy + h / 2;
  const eave = cy - h / 6;
  const apex = cy - h / 2 - h / 3;
  return [
    `M ${left} ${bottom}`,
    `L ${left} ${eave}`,
    `L ${cx} ${apex}`,
    `L ${right} ${eave}`,
    `L ${right} ${bottom}`,
    "Z",
  ].join(" ");
}

export default function Building({ cx, cy, size, player, kind }: Props) {
  const s = kind === "city" ? size * 1.5 : size;
  return (
    <g className="piece-pop">
      <title>
        Player {player + 1} {kind}
      </title>
      <path
        d={housePath(cx, cy, s)}
        fill={PLAYER_COLORS[player]}
        stroke={PLAYER_STROKES[player]}
        strokeWidth={2}
        strokeLinejoin="round"
      />
    </g>
  );
}
