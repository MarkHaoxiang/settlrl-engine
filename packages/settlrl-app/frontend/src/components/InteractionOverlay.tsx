import { cubeToPixel, hexCorners, hexToPixel, type Cube, type CubeEdge, type Hex } from "../lib/hex";
import { PLAYER_COLORS } from "../lib/boardData";
import { HIGHLIGHT } from "../lib/ui";
import { housePath } from "./Building";

// A popover anchor in viewport coordinates (top-centre of the clicked element).
// Floating UI positions the popover against it and keeps it on-screen.
export interface BoardTargetPoint {
  x: number;
  y: number;
}

// Legal placement targets to mark on the board (for the Play view): the
// acting player's buildable spots as quiet markers that preview the full
// piece on hover, plus highlighted robber tiles. Handlers receive the clicked
// target and its anchor point so the caller can pop a chooser there.
export interface BoardInteraction {
  // Whose colour the markers and ghost previews borrow.
  player: number;
  vertices: { cube: Cube; kind: "settlement" | "city" }[];
  edges: CubeEdge[];
  tiles: Hex[];
  onVertex?: (vertex: Cube, at: BoardTargetPoint) => void;
  onEdge?: (edge: CubeEdge, at: BoardTargetPoint) => void;
  onTile?: (tile: Hex, at: BoardTargetPoint) => void;
}

interface Props {
  interaction: BoardInteraction;
  offsetX: number;
  offsetY: number;
  hex: number;
  anchorOf: (el: SVGGraphicsElement) => BoardTargetPoint;
}

// The clickable layer over the board. Marker vs hover-ghost styling lives in
// index.css (.board-ghost / .ghost-min / .ghost-full).
export default function InteractionOverlay({ interaction, offsetX, offsetY, hex, anchorOf }: Props) {
  const colour = PLAYER_COLORS[interaction.player];
  return (
    <g>
      {/* Legal robber tiles: a translucent hex with a ring */}
      {interaction.tiles.map((tile, i) => {
        const { x, y } = hexToPixel(tile, hex);
        const pts = hexCorners(x + offsetX, y + offsetY, hex * 0.94)
          .map(([px, py]) => `${px},${py}`)
          .join(" ");
        return (
          <g key={`itile-${i}`} className="board-ghost" onClick={(e) => interaction.onTile?.(tile, anchorOf(e.currentTarget))}>
            <polygon className="ghost" points={pts} fill={HIGHLIGHT} fillOpacity={0.3} stroke={HIGHLIGHT} strokeWidth={3} />
          </g>
        );
      })}

      {/* Road slots on legal edges: a short quiet centre dash, the full ghost
          road on hover (with a fat transparent hit line) */}
      {interaction.edges.map((edge, i) => {
        const a = cubeToPixel(edge.a, hex);
        const b = cubeToPixel(edge.b, hex);
        const line = { x1: a.x + offsetX, y1: a.y + offsetY, x2: b.x + offsetX, y2: b.y + offsetY };
        const lerp = (t: number) => ({
          x: line.x1 + (line.x2 - line.x1) * t,
          y: line.y1 + (line.y2 - line.y1) * t,
        });
        const m1 = lerp(0.32);
        const m2 = lerp(0.68);
        return (
          <g key={`iedge-${i}`} className="board-ghost" onClick={(e) => interaction.onEdge?.(edge, anchorOf(e.currentTarget))}>
            <line
              className="ghost-min"
              x1={m1.x}
              y1={m1.y}
              x2={m2.x}
              y2={m2.y}
              stroke={colour}
              strokeWidth={hex * 0.07}
              strokeLinecap="round"
            />
            <line
              className="ghost-full"
              {...line}
              stroke={colour}
              strokeWidth={hex * 0.14}
              strokeDasharray="8 5"
              strokeLinecap="round"
            />
            <line {...line} stroke="transparent" strokeWidth={hex * 0.3} strokeLinecap="round" />
          </g>
        );
      })}

      {/* Building slots on legal vertices: a small quiet dot, the full ghost
          piece on hover (settlement house, or the larger city outline over an
          upgradable settlement) */}
      {interaction.vertices.map(({ cube, kind }, i) => {
        const { x, y } = cubeToPixel(cube, hex);
        const s = hex * 0.3 * (kind === "city" ? 1.5 : 1);
        return (
          <g key={`ivert-${i}`} className="board-ghost" onClick={(e) => interaction.onVertex?.(cube, anchorOf(e.currentTarget))}>
            <circle
              className="ghost-min"
              cx={x + offsetX}
              cy={y + offsetY}
              r={hex * (kind === "city" ? 0.12 : 0.09)}
              fill={colour}
              stroke={HIGHLIGHT}
              strokeWidth={1.5}
            />
            <path
              className="ghost-full"
              d={housePath(x + offsetX, y + offsetY, s)}
              fill={colour}
              fillOpacity={0.5}
              stroke={HIGHLIGHT}
              strokeWidth={2}
              strokeDasharray="5 3"
              strokeLinejoin="round"
            />
            <circle cx={x + offsetX} cy={y + offsetY} r={hex * 0.3} fill="transparent" />
          </g>
        );
      })}
    </g>
  );
}
