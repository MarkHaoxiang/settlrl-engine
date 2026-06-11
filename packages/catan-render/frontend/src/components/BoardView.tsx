import { useEffect, useRef, useState } from "react";
import { hexToPixel, cubeToPixel, hexCorners, type Cube, type CubeEdge, type Hex } from "../lib/hex";
import { PLAYER_COLORS, type Board } from "../lib/boardData";
import { HIGHLIGHT } from "../lib/ui";
import HexTile from "./HexTile";
import Road from "./Road";
import Building, { housePath } from "./Building";
import Robber from "./Robber";
import Port from "./Port";
import BankStacks from "./BankStacks";
import PlayerAreas from "./PlayerAreas";
import TableDice from "./TableDice";

// A popover anchor in this component's container coordinates (top-centre of
// the clicked element, valid for the pan/zoom at click time).
export interface BoardTargetPoint {
  x: number;
  y: number;
}

// Legal placement targets to ghost onto the board (for the Play view): the
// acting player's buildable spots drawn as faint pieces in their colour, plus
// highlighted robber tiles. Handlers receive the clicked target and its anchor
// point so the caller can pop a chooser there.
export interface BoardInteraction {
  // Whose colour the ghost previews borrow.
  player: number;
  vertices: { cube: Cube; kind: "settlement" | "city" }[];
  edges: CubeEdge[];
  tiles: Hex[];
  onVertex?: (vertex: Cube, at: BoardTargetPoint) => void;
  onEdge?: (edge: CubeEdge, at: BoardTargetPoint) => void;
  onTile?: (tile: Hex, at: BoardTargetPoint) => void;
}

const HEX_SIZE = 72;
const PADDING = 90;

// Physical table scale: a real hex is 80mm flat-to-flat and a card 57×89mm,
// so cards render true to size against the tiles.
const CARD_W = (Math.sqrt(3) * HEX_SIZE * 57) / 80;
const CARD_H = (Math.sqrt(3) * HEX_SIZE * 89) / 80;
// The table band around the ocean holding each seat's play area.
const EDGE_BAND = CARD_H + 44;

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// The table dice: the last roll's sum, a per-move seed for their resting
// angles, and the roll handler while rolling is the viewer's move.
export interface DiceState {
  sum: number;
  seed: number;
  onRoll?: () => void;
}

interface Props {
  board: Board;
  interaction?: BoardInteraction;
  dice?: DiceState;
}

// Renders a Catan board (tiles, ports, roads, buildings, robber, the bank's
// card stacks on the table beside it) as a zoomable, pannable SVG. It fills
// its parent container, so a parent can overlay mode-specific controls on top
// (the replay scrubber, the play action bar, a back button, …).
export default function BoardView({ board, interaction, dice }: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Active drag-to-pan gesture; `moved` flips once past the click-vs-drag
  // threshold, after which the pointer is captured so releasing over a board
  // element doesn't also click it.
  const drag = useRef<{
    id: number;
    x: number;
    y: number;
    panX: number;
    panY: number;
    moved: boolean;
  } | null>(null);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (drag.current) {
      // A second touch landed: this is a pinch, not a pan.
      drag.current = null;
      return;
    }
    drag.current = {
      id: e.pointerId,
      x: e.clientX,
      y: e.clientY,
      panX: pan.x,
      panY: pan.y,
      moved: false,
    };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || e.pointerId !== d.id) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) > 4) {
      d.moved = true;
      e.currentTarget.setPointerCapture(d.id);
    }
    if (d.moved) setPan({ x: d.panX + dx, y: d.panY + dy });
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (drag.current?.id === e.pointerId) drag.current = null;
  };

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

  // Anchor for the caller's popover: top-centre of the clicked SVG element,
  // converted to this container's coordinate space.
  const anchorOf = (el: SVGGraphicsElement): BoardTargetPoint => {
    const c = containerRef.current!.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2 - c.x, y: r.y - c.y };
  };

  const pixels = board.tiles.map((t) => hexToPixel(t.hex, HEX_SIZE));


  const minX = Math.min(...pixels.map((p) => p.x));
  const maxX = Math.max(...pixels.map((p) => p.x));
  const minY = Math.min(...pixels.map((p) => p.y));
  const maxY = Math.max(...pixels.map((p) => p.y));

  // The scene is a tabletop: the ocean board in the middle, a band around it
  // holding each seat's play area, and a wider band left of that for the
  // bank's card grid. Everything lives in board coordinates, so it all pans
  // and zooms together.
  const oceanW = maxX - minX + HEX_SIZE * 2 + PADDING * 2;
  const oceanH = maxY - minY + HEX_SIZE * 2 + PADDING * 2;
  const bankBand = board.bank ? CARD_W * 2 + 60 : 0;
  const oceanX = bankBand + EDGE_BAND;
  const oceanY = EDGE_BAND;
  const width = oceanW + oceanX + EDGE_BAND;
  const height = oceanH + EDGE_BAND * 2;
  const offsetX = -minX + HEX_SIZE + PADDING + oceanX;
  const offsetY = -minY + HEX_SIZE + PADDING + oceanY;

  // Open with the whole table in view — the edge bands make the scene bigger
  // than the viewport at 1:1; wheel/pinch still zoom freely afterwards.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const fit = Math.min(el.clientWidth / width, el.clientHeight / height, 1);
    setZoom(clamp(fit * 0.98, MIN_ZOOM, 1));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  return (
    <div
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        touchAction: "none",
        cursor: "grab",
      }}
    >
      <div
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: "center center",
        }}
      >
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

          {/* Ocean board in the middle of the table */}
          <rect x={oceanX} y={oceanY} width={oceanW} height={oceanH} fill="url(#oceanGrad)" rx={24} />

          {/* The bank's card grid on the table, left of everyone */}
          {board.bank && (
            <BankStacks bank={board.bank} cx={bankBand / 2} cy={height / 2} cardW={CARD_W} cardH={CARD_H} />
          )}

          {/* The dice rest in the table's bottom-right corner */}
          {dice && (
            <TableDice
              cx={oceanX + oceanW + EDGE_BAND / 2}
              cy={oceanY + oceanH + EDGE_BAND / 2}
              size={HEX_SIZE * 0.4}
              sum={dice.sum}
              seed={dice.seed}
              onRoll={dice.onRoll}
            />
          )}

          {/* Each seat's play area on its table edge */}
          <PlayerAreas
            board={board}
            oceanX={oceanX}
            oceanY={oceanY}
            oceanW={oceanW}
            oceanH={oceanH}
            band={EDGE_BAND}
            cardW={CARD_W}
            cardH={CARD_H}
            hex={HEX_SIZE}
          />

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

          {/* Roads sit under buildings. Keyed by edge identity so existing
              pieces keep their DOM node and only newly placed ones mount
              (and play their pop-in animation). */}
          {board.roads.map((road) => {
            const a = cubeToPixel(road.a, HEX_SIZE);
            const b = cubeToPixel(road.b, HEX_SIZE);
            return (
              <Road
                key={`road-${road.a.q},${road.a.r},${road.a.s}-${road.b.q},${road.b.r},${road.b.s}`}
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

          {/* Settlements and cities sit on top. Keyed by vertex + kind: a city
              upgrade remounts the piece, so it pops in too. */}
          {board.buildings.map((b) => {
            const { x, y } = cubeToPixel(b.cube, HEX_SIZE);
            return (
              <Building
                key={`building-${b.cube.q},${b.cube.r},${b.cube.s}-${b.kind}`}
                cx={x + offsetX}
                cy={y + offsetY}
                size={HEX_SIZE * 0.3}
                player={b.player}
                kind={b.kind}
              />
            );
          })}

          {/* Interaction overlay: ghost previews of legal placements sit on top */}
          {interaction && (
            <g>
              {/* Legal robber tiles: a translucent hex with a ring */}
              {interaction.tiles.map((tile, i) => {
                const { x, y } = hexToPixel(tile, HEX_SIZE);
                const pts = hexCorners(x + offsetX, y + offsetY, HEX_SIZE * 0.94)
                  .map(([px, py]) => `${px},${py}`)
                  .join(" ");
                return (
                  <g
                    key={`itile-${i}`}
                    className="board-ghost"
                    onClick={(e) => interaction.onTile?.(tile, anchorOf(e.currentTarget))}
                  >
                    <polygon
                      className="ghost"
                      points={pts}
                      fill={HIGHLIGHT}
                      fillOpacity={0.3}
                      stroke={HIGHLIGHT}
                      strokeWidth={3}
                    />
                  </g>
                );
              })}

              {/* Road slots on legal edges: a short quiet centre dash, the
                  full ghost road on hover (with a fat transparent hit line) */}
              {interaction.edges.map((edge, i) => {
                const a = cubeToPixel(edge.a, HEX_SIZE);
                const b = cubeToPixel(edge.b, HEX_SIZE);
                const line = {
                  x1: a.x + offsetX,
                  y1: a.y + offsetY,
                  x2: b.x + offsetX,
                  y2: b.y + offsetY,
                };
                const lerp = (t: number) => ({
                  x: line.x1 + (line.x2 - line.x1) * t,
                  y: line.y1 + (line.y2 - line.y1) * t,
                });
                const m1 = lerp(0.32);
                const m2 = lerp(0.68);
                return (
                  <g
                    key={`iedge-${i}`}
                    className="board-ghost"
                    onClick={(e) => interaction.onEdge?.(edge, anchorOf(e.currentTarget))}
                  >
                    <line
                      className="ghost-min"
                      x1={m1.x}
                      y1={m1.y}
                      x2={m2.x}
                      y2={m2.y}
                      stroke={PLAYER_COLORS[interaction.player]}
                      strokeWidth={HEX_SIZE * 0.07}
                      strokeLinecap="round"
                    />
                    <line
                      className="ghost-full"
                      {...line}
                      stroke={PLAYER_COLORS[interaction.player]}
                      strokeWidth={HEX_SIZE * 0.14}
                      strokeDasharray="8 5"
                      strokeLinecap="round"
                    />
                    <line {...line} stroke="transparent" strokeWidth={HEX_SIZE * 0.3} strokeLinecap="round" />
                  </g>
                );
              })}

              {/* Building slots on legal vertices: a small quiet dot, the full
                  ghost piece on hover (settlement house, or the larger city
                  outline over an upgradable settlement) */}
              {interaction.vertices.map(({ cube, kind }, i) => {
                const { x, y } = cubeToPixel(cube, HEX_SIZE);
                const s = HEX_SIZE * 0.3 * (kind === "city" ? 1.5 : 1);
                return (
                  <g
                    key={`ivert-${i}`}
                    className="board-ghost"
                    onClick={(e) => interaction.onVertex?.(cube, anchorOf(e.currentTarget))}
                  >
                    <circle
                      className="ghost-min"
                      cx={x + offsetX}
                      cy={y + offsetY}
                      r={HEX_SIZE * (kind === "city" ? 0.12 : 0.09)}
                      fill={PLAYER_COLORS[interaction.player]}
                      stroke={HIGHLIGHT}
                      strokeWidth={1.5}
                    />
                    <path
                      className="ghost-full"
                      d={housePath(x + offsetX, y + offsetY, s)}
                      fill={PLAYER_COLORS[interaction.player]}
                      fillOpacity={0.5}
                      stroke={HIGHLIGHT}
                      strokeWidth={2}
                      strokeDasharray="5 3"
                      strokeLinejoin="round"
                    />
                    <circle cx={x + offsetX} cy={y + offsetY} r={HEX_SIZE * 0.3} fill="transparent" />
                  </g>
                );
              })}
            </g>
          )}
        </svg>
      </div>

    </div>
  );
}
