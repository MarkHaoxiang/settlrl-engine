import type { ResourceKind } from "../lib/boardData";
import TerrainIcon from "./TerrainIcon";

// A resource's terrain motif as a standalone inline glyph, sized in pixels for
// a chip / badge / cost row. The motif is drawn in TerrainIcon's ±11 box, so
// `scale` ~1 fills the glyph; callers tweak it (and opacity) per context.
export default function ResourceGlyph({
  kind,
  px,
  scale = 1.05,
  opacity = 0.9,
}: {
  kind: ResourceKind;
  px: number;
  scale?: number;
  opacity?: number;
}) {
  return (
    <svg width={px} height={px} viewBox="-11 -11 22 22">
      <TerrainIcon terrain={kind} cx={0} cy={0} scale={scale} opacity={opacity} />
    </svg>
  );
}
