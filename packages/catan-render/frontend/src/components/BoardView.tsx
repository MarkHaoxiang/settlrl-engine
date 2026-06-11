import { hexToPixel, cubeToPixel } from "../lib/hex";
import type { Board, ResourceKind } from "../lib/boardData";
import { useTableViewport } from "../lib/viewport";
import { panelStyle } from "../lib/ui";
import HexTile from "./HexTile";
import Road from "./Road";
import Building from "./Building";
import Robber from "./Robber";
import Port from "./Port";
import BankStacks from "./BankStacks";
import PlayerAreas from "./PlayerAreas";
import TableDice from "./TableDice";
import InteractionOverlay, { type BoardInteraction, type BoardTargetPoint } from "./InteractionOverlay";

export type { BoardInteraction, BoardTargetPoint };

const HEX_SIZE = 72;
const PADDING = 90;

// Physical table scale: a real hex is 80mm flat-to-flat and a card 57×89mm,
// so cards render true to size against the tiles.
const CARD_W = (Math.sqrt(3) * HEX_SIZE * 57) / 80;
const CARD_H = (Math.sqrt(3) * HEX_SIZE * 89) / 80;
// The table band around the ocean holding each seat's play area.
const EDGE_BAND = CARD_H + 44;

// The table dice: the last roll's sum, a per-move seed for their resting
// angles, and the roll handler while rolling is the viewer's move.
export interface DiceState {
  sum: number;
  seed: number;
  onRoll?: () => void;
}

// Trade targets on the table: bank piles the viewer can trade for, and seats
// they can propose to (clicking an opponent's hand pile opens the offer).
export interface TradeTargets {
  bank: Set<ResourceKind>;
  partners: Set<number>;
  onBank: (r: ResourceKind, at: BoardTargetPoint) => void;
  onPartner: (p: number, at: BoardTargetPoint) => void;
}

interface Props {
  board: Board;
  interaction?: BoardInteraction;
  dice?: DiceState;
  trade?: TradeTargets;
}

// The whole table seen from above: the ocean board in the middle (tiles,
// ports, roads, buildings, robber), the bank's card grid left of it, each
// seat's play area on its table edge, and the dice in a corner — one SVG
// scene that pans, zooms, and spins together (lib/viewport.ts). It fills its
// parent container, so a parent can overlay mode-specific controls on top.
export default function BoardView({ board, interaction, dice, trade }: Props) {
  const pixels = board.tiles.map((t) => hexToPixel(t.hex, HEX_SIZE));
  const minX = Math.min(...pixels.map((p) => p.x));
  const maxX = Math.max(...pixels.map((p) => p.x));
  const minY = Math.min(...pixels.map((p) => p.y));
  const maxY = Math.max(...pixels.map((p) => p.y));

  const oceanW = maxX - minX + HEX_SIZE * 2 + PADDING * 2;
  const oceanH = maxY - minY + HEX_SIZE * 2 + PADDING * 2;
  const bankBand = board.bank ? CARD_W * 2 + 60 : 0;
  const oceanX = bankBand + EDGE_BAND;
  const oceanY = EDGE_BAND;
  const width = oceanW + oceanX + EDGE_BAND;
  const height = oceanH + EDGE_BAND * 2;
  const offsetX = -minX + HEX_SIZE + PADDING + oceanX;
  const offsetY = -minY + HEX_SIZE + PADDING + oceanY;

  const { containerRef, containerHandlers, sceneTransform, rotationTransform, rotate } =
    useTableViewport(width, height);

  // Anchor for the caller's popover: top-centre of the clicked SVG element,
  // converted to this container's coordinate space.
  const anchorOf = (el: SVGGraphicsElement): BoardTargetPoint => {
    const c = containerRef.current!.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2 - c.x, y: r.y - c.y };
  };

  return (
    <div
      ref={containerRef}
      {...containerHandlers}
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
      <div style={{ transform: sceneTransform, transformOrigin: "center center" }}>
        <div
          style={{
            transform: rotationTransform,
            transformOrigin: "center center",
            transition: "transform 0.45s ease",
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
              <BankStacks
                bank={board.bank}
                cx={bankBand / 2}
                cy={height / 2}
                cardW={CARD_W}
                cardH={CARD_H}
                tradable={trade?.bank}
                onPick={trade && ((r, el) => trade.onBank(r, anchorOf(el)))}
              />
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
              partners={trade?.partners}
              onPartner={trade && ((p, el) => trade.onPartner(p, anchorOf(el)))}
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
            {board.tiles.map((tile, i) => (
              <HexTile
                key={`tile-${i}`}
                cx={pixels[i].x + offsetX}
                cy={pixels[i].y + offsetY}
                size={HEX_SIZE}
                terrain={tile.terrain}
                number={tile.number}
              />
            ))}

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

            {/* Legal-placement markers sit on top */}
            {interaction && (
              <InteractionOverlay
                interaction={interaction}
                offsetX={offsetX}
                offsetY={offsetY}
                hex={HEX_SIZE}
                anchorOf={anchorOf}
              />
            )}
          </svg>
        </div>
      </div>

      {/* Spin the table a quarter turn at a time (e.g. to face your seat) */}
      <div
        style={{ position: "absolute", bottom: 16, left: 16, display: "flex", gap: 6, zIndex: 9 }}
        onPointerDown={(e) => e.stopPropagation()}
      >
        {(
          [
            ["↺", -90, "Rotate the table counter-clockwise ( [ )"],
            ["↻", 90, "Rotate the table clockwise ( ] )"],
          ] as const
        ).map(([glyph, step, label]) => (
          <button
            key={glyph}
            title={label}
            style={{
              ...panelStyle,
              width: 32,
              height: 32,
              borderRadius: "50%",
              fontSize: 16,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
            onClick={() => rotate(step)}
          >
            {glyph}
          </button>
        ))}
      </div>
    </div>
  );
}
