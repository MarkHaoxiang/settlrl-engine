import { hexToPixel } from "../lib/hex";
import { BOARD_TILES } from "../lib/boardData";
import HexTile from "./HexTile";

const HEX_SIZE = 72;
const PADDING = 90;

export default function CatanBoard() {
  const pixels = BOARD_TILES.map((t) => hexToPixel(t.hex, HEX_SIZE));

  const minX = Math.min(...pixels.map((p) => p.x));
  const maxX = Math.max(...pixels.map((p) => p.x));
  const minY = Math.min(...pixels.map((p) => p.y));
  const maxY = Math.max(...pixels.map((p) => p.y));

  const width = maxX - minX + HEX_SIZE * 2 + PADDING * 2;
  const height = maxY - minY + HEX_SIZE * 2 + PADDING * 2;
  const offsetX = -minX + HEX_SIZE + PADDING;
  const offsetY = -minY + HEX_SIZE + PADDING;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block", filter: "drop-shadow(0 8px 32px rgba(0,0,0,0.5))" }}
    >
      <defs>
        <radialGradient id="oceanGrad" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor="#2176AE" />
          <stop offset="100%" stopColor="#0D3B66" />
        </radialGradient>
      </defs>

      {/* Ocean background */}
      <rect width={width} height={height} fill="url(#oceanGrad)" rx={24} />

      {/* Tiles */}
      {BOARD_TILES.map((tile, i) => {
        const { x, y } = pixels[i];
        return (
          <HexTile
            key={i}
            cx={x + offsetX}
            cy={y + offsetY}
            size={HEX_SIZE}
            terrain={tile.terrain}
            number={tile.number}
          />
        );
      })}
    </svg>
  );
}
