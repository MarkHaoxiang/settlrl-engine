// Board geometry: the engine's axial/cube coordinates -> SVG pixels.
//
// Tiles are pointy-top hexes addressed by axial (q, r); board vertices use the
// engine's cube coordinates (q, r, s), where tile centres sum to 0 and corner
// vertices sum to ±1 (a vertex is one unit along a single axis from each of
// the three tiles that share it).

export interface Hex {
  q: number;
  r: number;
}

// A board vertex in the engine's cube coordinates (q + r + s = ±1).
export interface Cube {
  q: number;
  r: number;
  s: number;
}

// An edge between two vertices.
export interface CubeEdge {
  a: Cube;
  b: Cube;
}

// Axial → pixel, pointy-top orientation
export function hexToPixel(hex: Hex, size: number): { x: number; y: number } {
  return {
    x: size * (Math.sqrt(3) * hex.q + (Math.sqrt(3) / 2) * hex.r),
    y: size * (1.5 * hex.r),
  };
}

// Vertex cube → pixel. A vertex (cube sum ±1) is the corner shared by the
// three tiles at cube − sign·eᵢ (one per axis), i.e. the centroid of those
// tile centres; the projection is linear, so average in axial space first.
export function cubeToPixel(cube: Cube, size: number): { x: number; y: number } {
  const sign = cube.q + cube.r + cube.s; // +1 or -1
  return hexToPixel({ q: cube.q - sign / 3, r: cube.r - sign / 3 }, size);
}

// Six corner points of a pointy-top hex centred at (cx, cy)
export function hexCorners(
  cx: number,
  cy: number,
  size: number
): [number, number][] {
  return Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 180) * (60 * i - 30);
    return [cx + size * Math.cos(angle), cy + size * Math.sin(angle)];
  });
}

// -- coordinate equality ----------------------------------------------------

export const hexEq = (a: Hex, b: Hex): boolean => a.q === b.q && a.r === b.r;

export const cubeEq = (a: Cube, b: Cube): boolean =>
  a.q === b.q && a.r === b.r && a.s === b.s;

// Edges are undirected: (a, b) and (b, a) are the same edge.
export const edgeEq = (a: CubeEdge, b: CubeEdge): boolean =>
  (cubeEq(a.a, b.a) && cubeEq(a.b, b.b)) || (cubeEq(a.a, b.b) && cubeEq(a.b, b.a));
