import { TERRAIN_FILL, TERRAIN_STROKE, type Terrain } from "../lib/boardData";

interface Props {
  // Pixel positions of the two coastal vertices the port serves.
  ax: number;
  ay: number;
  bx: number;
  by: number;
  // Board centre, used to push the badge outward into the ocean.
  cx: number;
  cy: number;
  // 2:1 resource port, or null for a 3:1 general port.
  resource: Terrain | null;
  size: number;
}

const GENERAL_FILL = "#D9C7A3";
const GENERAL_STROKE = "#8A7A5C";
const DOCK_COLOR = "#6B4A2B";

export default function Port({ ax, ay, bx, by, cx, cy, resource, size }: Props) {
  const mx = (ax + bx) / 2;
  const my = (ay + by) / 2;

  // Outward (away from board centre) unit vector.
  const dx = mx - cx;
  const dy = my - cy;
  const len = Math.hypot(dx, dy) || 1;
  const nx = dx / len;
  const ny = dy / len;

  const badgeX = mx + nx * size * 0.6;
  const badgeY = my + ny * size * 0.6;
  const radius = size * 0.26;

  const fill = resource ? TERRAIN_FILL[resource] : GENERAL_FILL;
  const stroke = resource ? TERRAIN_STROKE[resource] : GENERAL_STROKE;
  const label = resource ? "2:1" : "3:1";

  return (
    <g>
      <title>{resource ? `${resource} 2:1 port` : "3:1 port"}</title>
      {/* Docks linking the badge to each coastal vertex */}
      <line x1={badgeX} y1={badgeY} x2={ax} y2={ay} stroke={DOCK_COLOR} strokeWidth={size * 0.07} strokeLinecap="round" />
      <line x1={badgeX} y1={badgeY} x2={bx} y2={by} stroke={DOCK_COLOR} strokeWidth={size * 0.07} strokeLinecap="round" />
      {/* Badge */}
      <circle cx={badgeX} cy={badgeY} r={radius} fill={fill} stroke={stroke} strokeWidth={2.5} />
      <text
        x={badgeX}
        y={badgeY}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={size * 0.2}
        fontWeight="bold"
        fill="#2C1A00"
        fontFamily="Georgia, 'Times New Roman', serif"
      >
        {label}
      </text>
    </g>
  );
}
