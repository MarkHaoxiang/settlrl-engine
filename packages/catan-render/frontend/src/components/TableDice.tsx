import { HIGHLIGHT } from "../lib/ui";

// Pip layout per face, in pip-grid offsets around the die centre.
const PIPS: Record<number, [number, number][]> = {
  1: [[0, 0]],
  2: [
    [-1, -1],
    [1, 1],
  ],
  3: [
    [-1, -1],
    [0, 0],
    [1, 1],
  ],
  4: [
    [-1, -1],
    [1, -1],
    [-1, 1],
    [1, 1],
  ],
  5: [
    [-1, -1],
    [1, -1],
    [0, 0],
    [-1, 1],
    [1, 1],
  ],
  6: [
    [-1, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [1, 1],
  ],
};

function Die({
  cx,
  cy,
  size,
  value,
  rot,
}: {
  cx: number;
  cy: number;
  size: number;
  value: number;
  rot: number;
}) {
  const half = size / 2;
  const off = size * 0.27;
  return (
    <g transform={`rotate(${rot} ${cx} ${cy})`}>
      <rect
        x={cx - half}
        y={cy - half}
        width={size}
        height={size}
        rx={size * 0.2}
        fill="#FDF6E3"
        stroke="#A08050"
        strokeWidth={1.5}
      />
      {(PIPS[value] ?? []).map(([px, py], i) => (
        <circle key={i} cx={cx + px * off} cy={cy + py * off} r={size * 0.09} fill="#2C1A00" />
      ))}
    </g>
  );
}

// Split a 2d6 sum into two plausible faces (the engine stores only the sum).
function split(sum: number): [number, number] {
  const d1 = Math.min(6, Math.max(1, Math.ceil(sum / 2)));
  return [d1, Math.min(6, Math.max(1, sum - d1))];
}

// Two dice resting on the table by the board's corner: they show the last
// roll (blank and dim before the first), and glow clickable when it's the
// viewer's turn to roll.
export default function TableDice({
  cx,
  cy,
  size,
  sum,
  seed,
  onRoll,
}: {
  cx: number;
  cy: number;
  size: number;
  sum: number;
  // Varies the dice's resting angles per move, so consecutive rolls move.
  seed: number;
  onRoll?: () => void;
}) {
  const [d1, d2] = sum >= 2 ? split(sum) : [0, 0];
  const r1 = ((seed * 53 + sum * 17) % 44) - 22;
  const r2 = ((seed * 31 + sum * 7) % 44) - 22;
  const gap = size * 0.75;
  return (
    <g onClick={onRoll} style={onRoll ? { cursor: "pointer" } : undefined} opacity={sum >= 2 || onRoll ? 1 : 0.45}>
      <title>{onRoll ? "Roll the dice" : sum >= 2 ? `last roll: ${sum}` : "dice"}</title>
      {onRoll && (
        <circle cx={cx} cy={cy} r={size * 1.7} fill={HIGHLIGHT} fillOpacity={0.25} stroke={HIGHLIGHT} strokeWidth={2}>
          <animate
            attributeName="r"
            values={`${size * 1.55};${size * 1.8};${size * 1.55}`}
            dur="1.4s"
            repeatCount="indefinite"
          />
        </circle>
      )}
      <Die cx={cx - gap} cy={cy} size={size} value={d1} rot={r1} />
      <Die cx={cx + gap} cy={cy + size * 0.1} size={size} value={d2} rot={r2} />
    </g>
  );
}
