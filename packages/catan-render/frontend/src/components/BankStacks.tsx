import {
  TERRAIN_FILL,
  TERRAIN_STROKE,
  type Bank,
  type ResourceKind,
} from "../lib/boardData";
import TerrainIcon from "./TerrainIcon";

const DEV_FILL = "#5B4B8A";
const DEV_STROKE = "#3C3160";

// One face-up pile of cards seen from above: a couple of slightly skewed
// cards underneath, the top card carrying the icon and a count token. An
// exhausted pile leaves a dashed empty slot.
function Stack({
  cx,
  cy,
  w,
  h,
  count,
  fill,
  stroke,
  label,
  icon,
}: {
  cx: number;
  cy: number;
  w: number;
  h: number;
  count: number;
  fill: string;
  stroke: string;
  label: string;
  icon: React.ReactNode;
}) {
  const card = { x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: 5 };
  return (
    <g>
      <title>{`${label}: ${count} left`}</title>
      {count === 0 ? (
        <rect {...card} fill="none" stroke={stroke} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.5} />
      ) : (
        <>
          {[-5, 3].map((deg) => (
            <rect
              key={deg}
              {...card}
              transform={`rotate(${deg} ${cx} ${cy})`}
              fill={fill}
              stroke={stroke}
              strokeWidth={1.5}
              opacity={0.8}
            />
          ))}
          <rect {...card} fill={fill} stroke={stroke} strokeWidth={1.5} />
          {icon}
        </>
      )}
      <circle cx={cx} cy={cy + h * 0.22} r={h * 0.17} fill="#FDF6E3" stroke="#A08050" strokeWidth={1.2} />
      <text
        x={cx}
        y={cy + h * 0.22}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={h * 0.2}
        fontWeight="bold"
        fill="#2C1A00"
        fontFamily="Georgia, 'Times New Roman', serif"
      >
        {count}
      </text>
    </g>
  );
}

const ORDER: ResourceKind[] = ["wood", "brick", "sheep", "wheat", "ore"];

// The bank as physical card piles on the table beside the board: one per
// resource plus the development deck, drawn in board space (SVG) so they pan
// and zoom with everything else.
export default function BankStacks({
  bank,
  cx,
  cy,
  size,
}: {
  bank: Bank;
  cx: number;
  // Column centre; stacks spread vertically around it.
  cy: number;
  // HEX_SIZE: the card dimensions derive from it.
  size: number;
}) {
  const w = size * 0.58;
  const h = size * 0.84;
  const step = h + size * 0.18;
  const slots = ORDER.length + 1;
  const top = cy - ((slots - 1) / 2) * step;
  return (
    <g>
      {ORDER.map((r, i) => (
        <Stack
          key={r}
          cx={cx}
          cy={top + i * step}
          w={w}
          h={h}
          count={bank.resources[r]}
          fill={TERRAIN_FILL[r]}
          stroke={TERRAIN_STROKE[r]}
          label={r}
          icon={
            <TerrainIcon terrain={r} cx={cx} cy={top + i * step - h * 0.18} scale={size * 0.0095} opacity={0.85} />
          }
        />
      ))}
      <Stack
        cx={cx}
        cy={top + ORDER.length * step}
        w={w}
        h={h}
        count={bank.devCards}
        fill={DEV_FILL}
        stroke={DEV_STROKE}
        label="development cards"
        icon={
          <text
            x={cx}
            y={top + ORDER.length * step - h * 0.18}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={h * 0.3}
          >
            🃏
          </text>
        }
      />
    </g>
  );
}
