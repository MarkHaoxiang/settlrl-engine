import { ACCENT, HIGHLIGHT } from "../lib/ui";

interface Props {
  cx: number;
  cy: number;
  size: number;
  // While the viewer holds a playable knight, the pawn becomes a click target:
  // `armable` rings it as a hint, clicking arms knight targeting (`armed`), and
  // the legal destination tiles light up like a forced robber move.
  armable?: boolean;
  armed?: boolean;
  onClick?: () => void;
}

// A simple dark pawn marking the tile the robber occupies. Offset slightly off
// the tile centre so it doesn't fully cover the number token. The pawn is drawn
// around the origin and positioned with a CSS transform, so a robber move
// slides it to the new tile.
export default function Robber({ cx, cy, size, armable, armed, onClick }: Props) {
  const x = -size * 0.42;
  const w = size * 0.32;
  const h = size * 0.62;
  const interactive = armable && onClick;
  return (
    <g
      className={interactive ? "board-ghost" : undefined}
      onClick={interactive ? onClick : undefined}
      style={{ transform: `translate(${cx}px, ${cy}px)`, transition: "transform 0.45s ease" }}
    >
      <title>{armable ? "Play knight — click to move the robber" : "Robber"}</title>
      {/* Armed: a steady accent ring. Armable: a quiet highlight ring that
          brightens on hover (the .ghost class, shared with board markers). */}
      {(armable || armed) && (
        <circle
          className={armed ? undefined : "ghost"}
          cx={x}
          cy={-h * 0.2}
          r={size * 0.52}
          fill="none"
          stroke={armed ? ACCENT : HIGHLIGHT}
          strokeWidth={armed ? 4 : 3}
        />
      )}
      {/* Rounded body */}
      <rect
        x={x - w / 2}
        y={-h / 2}
        width={w}
        height={h}
        rx={w / 2}
        fill="#2B2B2B"
        stroke="#000000"
        strokeWidth={1.5}
      />
      {/* Head */}
      <circle cx={x} cy={-h / 2} r={w * 0.55} fill="#2B2B2B" stroke="#000000" strokeWidth={1.5} />
      {/* A generous transparent hit area so the small pawn is easy to click */}
      {interactive && <circle cx={x} cy={-h * 0.2} r={size * 0.55} fill="transparent" />}
    </g>
  );
}
