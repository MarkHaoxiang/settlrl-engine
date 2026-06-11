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
  const card = { x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: w * 0.09 };
  return (
    <g>
      <title>{`${label}: ${count} left`}</title>
      {count === 0 ? (
        <rect {...card} fill="none" stroke={stroke} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.5} />
      ) : (
        <>
          {[-3, 2].map((deg) => (
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
      <circle cx={cx} cy={cy + h * 0.26} r={h * 0.13} fill="#FDF6E3" stroke="#A08050" strokeWidth={1.2} />
      <text
        x={cx}
        y={cy + h * 0.26}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={h * 0.14}
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

// The bank as physical card piles on the table beside the board (cards at
// table scale): a two-column grid, one pile per resource plus the development
// deck, drawn in board space so they pan and zoom with everything else.
export default function BankStacks({
  bank,
  cx,
  cy,
  cardW,
  cardH,
}: {
  bank: Bank;
  // Grid centre.
  cx: number;
  cy: number;
  cardW: number;
  cardH: number;
}) {
  const stepX = cardW + 18;
  const stepY = cardH + 20;
  const at = (i: number) => ({
    x: cx + ((i % 2) - 0.5) * stepX,
    y: cy + (Math.floor(i / 2) - 1) * stepY,
  });
  return (
    <g>
      {ORDER.map((r, i) => {
        const { x, y } = at(i);
        return (
          <Stack
            key={r}
            cx={x}
            cy={y}
            w={cardW}
            h={cardH}
            count={bank.resources[r]}
            fill={TERRAIN_FILL[r]}
            stroke={TERRAIN_STROKE[r]}
            label={r}
            icon={<TerrainIcon terrain={r} cx={x} cy={y - cardH * 0.16} scale={cardW * 0.016} opacity={0.85} />}
          />
        );
      })}
      {(() => {
        const { x, y } = at(ORDER.length);
        return (
          <Stack
            cx={x}
            cy={y}
            w={cardW}
            h={cardH}
            count={bank.devCards}
            fill={DEV_FILL}
            stroke={DEV_STROKE}
            label="development cards"
            icon={
              <path
                d={`M ${x} ${y - cardH * 0.3} l ${cardW * 0.14} ${cardH * 0.14} l ${-cardW * 0.14} ${cardH * 0.14} l ${-cardW * 0.14} ${-cardH * 0.14} z`}
                fill="none"
                stroke="#C9B7E8"
                strokeWidth={2.5}
                strokeLinejoin="round"
              />
            }
          />
        );
      })()}
    </g>
  );
}
