import { useEffect, useRef, useState } from "react";
import { hexToPixel, cubeToPixel } from "../lib/hex";
import type { Board } from "../lib/boardData";
import HexTile from "./HexTile";
import Road from "./Road";
import Building from "./Building";
import Robber from "./Robber";
import Port from "./Port";
import PlayerPanel from "./PlayerPanel";

const HEX_SIZE = 72;
const PADDING = 90;

const MIN_ZOOM = 0.4;
const MAX_ZOOM = 3;

// Players are assigned to corners 0..3 = TL, TR, BL, BR.
const CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"] as const;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

interface Props {
  board: Board;
}

// Renders a Catan board (tiles, ports, roads, buildings, robber) as a zoomable
// SVG, with per-player stat panels anchored to the viewport corners. It fills
// its parent container, so a parent can overlay mode-specific controls on top
// (the replay scrubber, the play action bar, a back button, …).
export default function BoardView({ board }: Props) {
  const [zoom, setZoom] = useState(1);

  const containerRef = useRef<HTMLDivElement>(null);
  // Distance between the two active touch points during a pinch gesture.
  const pinchStart = useRef<{ dist: number; zoom: number } | null>(null);
  // Latest zoom, so the native handlers below read it without re-subscribing.
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;

  // Wheel + pinch zoom. Attached natively so we can preventDefault (React's
  // onWheel is passive and would still scroll/zoom the page).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // Trackpads report small deltas; mice report large ones. Exponential
      // scaling keeps the zoom feel consistent across both.
      const factor = Math.exp(-e.deltaY * 0.0015);
      setZoom((z) => clamp(z * factor, MIN_ZOOM, MAX_ZOOM));
    };

    const touchDist = (t: TouchList) =>
      Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        pinchStart.current = { dist: touchDist(e.touches), zoom: zoomRef.current };
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && pinchStart.current) {
        e.preventDefault();
        const ratio = touchDist(e.touches) / pinchStart.current.dist;
        setZoom(clamp(pinchStart.current.zoom * ratio, MIN_ZOOM, MAX_ZOOM));
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2) pinchStart.current = null;
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("touchstart", onTouchStart, { passive: false });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
    };
  }, []);

  const pixels = board.tiles.map((t) => hexToPixel(t.hex, HEX_SIZE));

  const minX = Math.min(...pixels.map((p) => p.x));
  const maxX = Math.max(...pixels.map((p) => p.x));
  const minY = Math.min(...pixels.map((p) => p.y));
  const maxY = Math.max(...pixels.map((p) => p.y));

  const width = maxX - minX + HEX_SIZE * 2 + PADDING * 2;
  const height = maxY - minY + HEX_SIZE * 2 + PADDING * 2;
  const offsetX = -minX + HEX_SIZE + PADDING;
  const offsetY = -minY + HEX_SIZE + PADDING;

  return (
    <div
      ref={containerRef}
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        touchAction: "none",
      }}
    >
      <div style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}>
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

          {/* Ports sit in the ocean; drawn first so docks tuck under the coast */}
          {board.ports.map((port, i) => {
            const a = cubeToPixel(port.a, HEX_SIZE);
            const b = cubeToPixel(port.b, HEX_SIZE);
            return (
              <Port
                key={`port-${i}`}
                ax={a.x + offsetX}
                ay={a.y + offsetY}
                bx={b.x + offsetX}
                by={b.y + offsetY}
                cx={offsetX}
                cy={offsetY}
                resource={port.resource}
                size={HEX_SIZE}
              />
            );
          })}

          {/* Tiles */}
          {board.tiles.map((tile, i) => {
            const { x, y } = pixels[i];
            return (
              <HexTile
                key={`tile-${i}`}
                cx={x + offsetX}
                cy={y + offsetY}
                size={HEX_SIZE}
                terrain={tile.terrain}
                number={tile.number}
              />
            );
          })}

          {/* Roads sit under buildings */}
          {board.roads.map((road, i) => {
            const a = cubeToPixel(road.a, HEX_SIZE);
            const b = cubeToPixel(road.b, HEX_SIZE);
            return (
              <Road
                key={`road-${i}`}
                x1={a.x + offsetX}
                y1={a.y + offsetY}
                x2={b.x + offsetX}
                y2={b.y + offsetY}
                player={road.player}
                width={HEX_SIZE * 0.14}
              />
            );
          })}

          {/* Robber */}
          {board.robber &&
            (() => {
              const { x, y } = hexToPixel(board.robber, HEX_SIZE);
              return <Robber cx={x + offsetX} cy={y + offsetY} size={HEX_SIZE} />;
            })()}

          {/* Settlements and cities sit on top */}
          {board.buildings.map((b, i) => {
            const { x, y } = cubeToPixel(b.cube, HEX_SIZE);
            return (
              <Building
                key={`building-${i}`}
                cx={x + offsetX}
                cy={y + offsetY}
                size={HEX_SIZE * 0.3}
                player={b.player}
                kind={b.kind}
              />
            );
          })}
        </svg>
      </div>

      {/* Corner player panels — fixed in the viewport, unaffected by zoom */}
      {board.players.map((p) => (
        <PlayerPanel key={`player-${p.player}`} player={p} corner={CORNERS[p.player]} />
      ))}
    </div>
  );
}
