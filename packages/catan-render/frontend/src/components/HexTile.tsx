import { hexCorners } from "../lib/hex";
import type { Terrain } from "../lib/boardData";

const FILL: Record<Terrain, string> = {
  wheat: "#EEC900",
  sheep: "#7DC95E",
  wood: "#2D6A2D",
  ore: "#8B949E",
  brick: "#C0392B",
  desert: "#E8D5A3",
};

const STROKE: Record<Terrain, string> = {
  wheat: "#B89A00",
  sheep: "#4E9A35",
  wood: "#1A4A1A",
  ore: "#5A666E",
  brick: "#8E2319",
  desert: "#C4B080",
};

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
