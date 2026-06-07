import { hexCorners } from "../lib/hex";
import { TERRAIN_FILL as FILL, TERRAIN_STROKE as STROKE } from "../lib/boardData";
import type { Terrain } from "../lib/boardData";
import TerrainIcon from "./TerrainIcon";

// Where the terrain motifs sit (fractions of the hex size from the centre):
// one above and two below the number token.
const MOTIF_SPOTS: [number, number][] = [
  [0, -0.52],
  [-0.45, 0.3],
  [0.45, 0.3],
];

interface Props {
  cx: number;
  cy: number;
  size: number;
  terrain: Terrain;
  number?: number;
}

function probDotCount(n: number): number {
  return 6 - Math.abs(7 - n);
}

export default function HexTile({ cx, cy, size, terrain, number }: Props) {
  const corners = hexCorners(cx, cy, size);
  const outerPoints = corners.map(([x, y]) => `${x},${y}`).join(" ");
  const innerCorners = hexCorners(cx, cy, size * 0.93);
  const innerPoints = innerCorners.map(([x, y]) => `${x},${y}`).join(" ");

  const isRed = number === 6 || number === 8;
  const tokenColor = isRed ? "#CC2200" : "#2C1A00";
  const dotCount = number !== undefined ? probDotCount(number) : 0;
  const dotSpacing = 5;
  const dotsWidth = (dotCount - 1) * dotSpacing;

  return (
    <g>
      <polygon
        points={outerPoints}
        fill={FILL[terrain]}
        stroke={STROKE[terrain]}
        strokeWidth={2.5}
      />
      <polygon
        points={innerPoints}
        fill="none"
        stroke={STROKE[terrain]}
        strokeWidth={1}
        opacity={0.4}
      />
      {MOTIF_SPOTS.map(([dx, dy], i) => (
        <TerrainIcon
          key={i}
          terrain={terrain}
          cx={cx + dx * size}
          cy={cy + dy * size}
          scale={size * 0.013}
        />
      ))}
      {number !== undefined && (
        <g>
          <circle
            cx={cx}
            cy={cy}
            r={size * 0.23}
            fill="#FDF6E3"
            stroke="#A08050"
            strokeWidth={1.5}
          />
          <text
            x={cx}
            y={cy - 2}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={size * 0.22}
            fontWeight="bold"
            fill={tokenColor}
            fontFamily="Georgia, 'Times New Roman', serif"
          >
            {number}
          </text>
          {/* Probability dots */}
          {Array.from({ length: dotCount }, (_, i) => (
            <circle
              key={i}
              cx={cx - dotsWidth / 2 + i * dotSpacing}
              cy={cy + size * 0.13}
              r={1.8}
              fill={tokenColor}
            />
          ))}
        </g>
      )}
    </g>
  );
}
