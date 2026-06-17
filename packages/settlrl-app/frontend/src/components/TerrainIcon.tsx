import type { ReactNode } from "react";
import { TERRAIN_STROKE } from "../lib/boardData";
import type { Terrain } from "../lib/boardData";

// A silhouette motif identifying a terrain (pine, brick wall, sheep, wheat
// ear, mountains, cactus), drawn around the origin in a roughly ±9-unit box.
// HexTile scatters a few per tile so terrains read by shape as well as colour.

const MOTIFS: Record<Terrain, ReactNode> = {
  wood: (
    <>
      <rect x={-1.2} y={4} width={2.4} height={4.5} />
      <polygon points="-7,5 7,5 0,-3" />
      <polygon points="-5,-0.5 5,-0.5 0,-8.5" />
    </>
  ),
  brick: (
    <>
      <rect x={-7.2} y={-5.4} width={6.8} height={3.2} rx={0.6} />
      <rect x={0.4} y={-5.4} width={6.8} height={3.2} rx={0.6} />
      <rect x={-3.4} y={-1.8} width={6.8} height={3.2} rx={0.6} />
      <rect x={-7.2} y={1.8} width={6.8} height={3.2} rx={0.6} />
      <rect x={0.4} y={1.8} width={6.8} height={3.2} rx={0.6} />
    </>
  ),
  sheep: (
    <>
      <rect x={-3.8} y={2.5} width={1.6} height={4.5} rx={0.8} />
      <rect x={1.8} y={2.5} width={1.6} height={4.5} rx={0.8} />
      <ellipse cx={0} cy={0} rx={5.8} ry={4} />
      <circle cx={5.6} cy={-2.6} r={2.4} />
    </>
  ),
  wheat: (
    <>
      <rect x={-0.7} y={-2.5} width={1.4} height={11.5} rx={0.7} />
      <ellipse cx={-2.2} cy={-2.5} rx={1.5} ry={2.8} transform="rotate(-40 -2.2 -2.5)" />
      <ellipse cx={2.2} cy={-2.5} rx={1.5} ry={2.8} transform="rotate(40 2.2 -2.5)" />
      <ellipse cx={-2.2} cy={-5.8} rx={1.5} ry={2.8} transform="rotate(-40 -2.2 -5.8)" />
      <ellipse cx={2.2} cy={-5.8} rx={1.5} ry={2.8} transform="rotate(40 2.2 -5.8)" />
      <ellipse cx={0} cy={-8.2} rx={1.5} ry={2.8} />
    </>
  ),
  ore: (
    <>
      <polygon points="-9,7 -2.5,-5 4,7" />
      <polygon points="1,7 5.8,-1 9.5,7" />
    </>
  ),
  desert: (
    <>
      <rect x={-1.6} y={-9} width={3.2} height={17} rx={1.6} />
      <rect x={-6.8} y={-6} width={2.8} height={6.5} rx={1.4} />
      <rect x={-6.8} y={-2.2} width={5.6} height={2.6} rx={1.3} />
      <rect x={4} y={-7.5} width={2.8} height={6} rx={1.4} />
      <rect x={1.2} y={-4.2} width={5.6} height={2.6} rx={1.3} />
    </>
  ),
};

interface Props {
  terrain: Terrain;
  cx: number;
  cy: number;
  scale: number;
  opacity?: number;
}

export default function TerrainIcon({ terrain, cx, cy, scale, opacity = 0.5 }: Props) {
  return (
    <g
      transform={`translate(${cx} ${cy}) scale(${scale})`}
      fill={TERRAIN_STROKE[terrain]}
      opacity={opacity}
    >
      {MOTIFS[terrain]}
    </g>
  );
}
