interface Props {
  cx: number;
  cy: number;
  size: number;
}

// A simple dark pawn marking the tile the robber occupies. Offset slightly off
// the tile centre so it doesn't fully cover the number token.
export default function Robber({ cx, cy, size }: Props) {
  const x = cx - size * 0.42;
  const w = size * 0.32;
  const h = size * 0.62;
  return (
    <g>
      <title>Robber</title>
      {/* Rounded body */}
      <rect
        x={x - w / 2}
        y={cy - h / 2}
        width={w}
        height={h}
        rx={w / 2}
        fill="#2B2B2B"
        stroke="#000000"
        strokeWidth={1.5}
      />
      {/* Head */}
      <circle
        cx={x}
        cy={cy - h / 2}
        r={w * 0.55}
        fill="#2B2B2B"
        stroke="#000000"
        strokeWidth={1.5}
      />
    </g>
  );
}
