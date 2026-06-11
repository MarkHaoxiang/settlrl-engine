import { hexCorners } from "../lib/hex";
import { TERRAIN_FILL as FILL, TERRAIN_STROKE as STROKE } from "../lib/boardData";
import type { Terrain } from "../lib/boardData";
import TerrainIcon from "./TerrainIcon";

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

// A clean tile: flat terrain colour, with one token integrating the terrain
// icon, the number, and its probability pips. The desert (no number) carries
// just a faint centred icon.
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
      {number === undefined ? (
        <TerrainIcon terrain={terrain} cx={cx} cy={cy} scale={size * 0.016} opacity={0.45} />
      ) : (
        <g>
          <circle
            cx={cx}
            cy={cy}
            r={size * 0.3}
            fill="#FDF6E3"
            stroke="#A08050"
            strokeWidth={1.5}
          />
          <TerrainIcon
            terrain={terrain}
            cx={cx}
            cy={cy - size * 0.15}
            scale={size * 0.0085}
            opacity={0.8}
          />
          <text
            x={cx}
            y={cy + size * 0.07}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={size * 0.21}
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
              cy={cy + size * 0.215}
              r={1.8}
              fill={tokenColor}
            />
          ))}
        </g>
      )}
    </g>
  );
}
