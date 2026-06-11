// Top-down card piles shared by the table scene (the bank's face-up stacks,
// each seat's face-down hand and dev piles).

// The cream number token that sits on piles.
export function CountToken({ cx, cy, r, value }: { cx: number; cy: number; r: number; value: number }) {
  return (
    <>
      <circle cx={cx} cy={cy} r={r} fill="#FDF6E3" stroke="#A08050" strokeWidth={1.2} />
      <text
        x={cx}
        y={cy}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={r * 1.1}
        fontWeight="bold"
        fill="#2C1A00"
        fontFamily="Georgia, 'Times New Roman', serif"
      >
        {value}
      </text>
    </>
  );
}

// A pile of cards seen from above: skewed cards underneath, a top card
// carrying `children` (face icon, back decoration), and a count token. An
// exhausted pile either leaves a dashed empty slot or disappears entirely.
export default function CardPile({
  cx,
  cy,
  w,
  h,
  count,
  fill,
  stroke,
  label,
  children,
  underlays = [-3, 2],
  empty = "slot",
}: {
  cx: number;
  cy: number;
  w: number;
  h: number;
  count: number;
  fill: string;
  stroke: string;
  label: string;
  children?: React.ReactNode;
  // Resting angles of the cards peeking out underneath the top card.
  underlays?: number[];
  empty?: "slot" | "hide";
}) {
  if (count === 0 && empty === "hide") return null;
  const card = { x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: w * 0.09 };
  return (
    <g>
      <title>{`${label}: ${count}`}</title>
      {count === 0 ? (
        <rect {...card} fill="none" stroke={stroke} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.5} />
      ) : (
        <>
          {underlays.map((deg) => (
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
          {children}
        </>
      )}
      <CountToken cx={cx} cy={cy + h * 0.26} r={h * 0.13} value={count} />
    </g>
  );
}
