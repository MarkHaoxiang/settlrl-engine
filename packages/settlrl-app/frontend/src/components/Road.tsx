import { PLAYER_COLORS, PLAYER_STROKES } from "../lib/boardData";

interface Props {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  player: number;
  width: number;
}

export default function Road({ x1, y1, x2, y2, player, width }: Props) {
  return (
    <g className="piece-pop">
      <title>Player {player + 1} road</title>
      {/* Darker casing for contrast against the tile edges. */}
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={PLAYER_STROKES[player]}
        strokeWidth={width + 3}
        strokeLinecap="round"
      />
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={PLAYER_COLORS[player]}
        strokeWidth={width}
        strokeLinecap="round"
      />
    </g>
  );
}
