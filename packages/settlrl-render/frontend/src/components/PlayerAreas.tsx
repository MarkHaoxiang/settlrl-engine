import {
  DEV_CARD_BACK,
  HAND_CARD_BACK,
  PLAYER_COLORS,
  PLAYER_STROKES,
  type Board,
  type Player,
} from "../lib/boardData";
import { HIGHLIGHT } from "../lib/ui";
import CardPile from "./CardPile";
import { housePath } from "./Building";

// Piece supply per player (base game).
const ROAD_SUPPLY = 15;
const SETTLEMENT_SUPPLY = 5;
const CITY_SUPPLY = 4;

// A face-down pile: a card back with an inner border and the owner's colour
// striped along its top edge. Gone entirely when empty (bare table).
function FaceDownPile({
  cx,
  cy,
  w,
  h,
  count,
  back,
  owner,
  label,
}: {
  cx: number;
  cy: number;
  w: number;
  h: number;
  count: number;
  back: { fill: string; stroke: string };
  owner: string;
  label: string;
}) {
  return (
    <CardPile
      cx={cx}
      cy={cy}
      w={w}
      h={h}
      count={count}
      fill={back.fill}
      stroke={back.stroke}
      label={label}
      underlays={count > 2 ? [-3] : []}
      empty="hide"
    >
      <rect
        x={cx - w * 0.32}
        y={cy - h * 0.32}
        width={w * 0.64}
        height={h * 0.64}
        rx={w * 0.06}
        fill="none"
        stroke={back.stroke}
        strokeWidth={1.2}
        opacity={0.6}
      />
      <rect x={cx - w / 2} y={cy - h / 2} width={w} height={h * 0.08} rx={w * 0.04} fill={owner} opacity={0.9} />
    </CardPile>
  );
}

// One seat's play area, laid out horizontally around the local origin (the
// parent rotates it onto its table edge): the face-down hand and dev piles,
// then the unbuilt supply — road sticks, settlement houses, city churches —
// one shape per piece still in the box.
function Area({
  player,
  roadsLeft,
  settlementsLeft,
  citiesLeft,
  cardW,
  cardH,
  hex,
  onHand,
}: {
  player: Player;
  roadsLeft: number;
  settlementsLeft: number;
  citiesLeft: number;
  cardW: number;
  cardH: number;
  hex: number;
  // Set while the viewer may propose this seat a trade: the hand pile
  // highlights and takes the click (reported with the element for anchoring).
  onHand?: (el: SVGGraphicsElement) => void;
}) {
  const color = PLAYER_COLORS[player.player] ?? "#888";
  const stroke = PLAYER_STROKES[player.player] ?? "#444";

  const roadLen = hex * 0.62;
  const roadW = hex * 0.075;
  const roadStep = roadW + 4;
  const settleSize = hex * 0.3;
  const settleStep = settleSize + 8;
  const citySize = hex * 0.42;
  const cityStep = citySize + 8;

  const pilesW = cardW * 2 + 16;
  const piecesW =
    ROAD_SUPPLY * roadStep + 26 + SETTLEMENT_SUPPLY * settleStep + 22 + CITY_SUPPLY * cityStep;
  let x = -(pilesW + 40 + piecesW) / 2;

  const handX = x + cardW / 2;
  const devX = x + cardW * 1.5 + 16;
  x += pilesW + 40;
  const roadsX = x;
  x += ROAD_SUPPLY * roadStep + 26;
  const settleX = x;
  x += SETTLEMENT_SUPPLY * settleStep + 22;
  const cityX = x;

  return (
    <g>
      {/* data-seat tags the resource hand pile as a fly-token endpoint, so a
          produced or stolen card lands on the pile itself (TransferAnimations). */}
      {onHand ? (
        <g data-seat={player.player} className="board-ghost" onClick={(e) => onHand(e.currentTarget)}>
          <rect
            className="ghost"
            x={handX - cardW / 2 - 6}
            y={-cardH / 2 - 6}
            width={cardW + 12}
            height={cardH + 12}
            rx={10}
            fill="none"
            stroke={HIGHLIGHT}
            strokeWidth={3}
          />
          <FaceDownPile cx={handX} cy={0} w={cardW} h={cardH} count={player.resourceCards} back={HAND_CARD_BACK} owner={color} label="resource cards — click to offer a trade" />
        </g>
      ) : (
        <g data-seat={player.player}>
          <FaceDownPile cx={handX} cy={0} w={cardW} h={cardH} count={player.resourceCards} back={HAND_CARD_BACK} owner={color} label="resource cards" />
        </g>
      )}
      <FaceDownPile cx={devX} cy={0} w={cardW} h={cardH} count={player.devCards} back={DEV_CARD_BACK} owner={color} label="development cards" />
      <g>
        <title>{`unbuilt roads: ${roadsLeft}`}</title>
        {Array.from({ length: roadsLeft }, (_, i) => (
          <rect
            key={i}
            x={roadsX + i * roadStep}
            y={-roadLen / 2}
            width={roadW}
            height={roadLen}
            rx={roadW / 2}
            fill={color}
            stroke={stroke}
            strokeWidth={1}
          />
        ))}
      </g>
      <g>
        <title>{`unbuilt settlements: ${settlementsLeft}`}</title>
        {Array.from({ length: settlementsLeft }, (_, i) => (
          <path
            key={i}
            d={housePath(settleX + i * settleStep + settleSize / 2, 0, settleSize)}
            fill={color}
            stroke={stroke}
            strokeWidth={1.5}
            strokeLinejoin="round"
          />
        ))}
      </g>
      <g>
        <title>{`unbuilt cities: ${citiesLeft}`}</title>
        {Array.from({ length: citiesLeft }, (_, i) => (
          <path
            key={i}
            d={housePath(cityX + i * cityStep + citySize / 2, 0, citySize)}
            fill={color}
            stroke={stroke}
            strokeWidth={1.5}
            strokeLinejoin="round"
          />
        ))}
      </g>
    </g>
  );
}

// Seats around the table in playing order, clockwise from the bottom edge
// (you): bottom, left, top, right — two-player games face each other. Each
// area is rotated to face its player, like looking down at a real table.
const EDGES_4 = ["bottom", "left", "top", "right"] as const;
const EDGES_2 = ["bottom", "top"] as const;
export type Edge = (typeof EDGES_4)[number];

const EDGE_ANGLE: Record<Edge, number> = { bottom: 0, left: 90, top: 180, right: -90 };

// The table edge a seat sits on, shared with the dice placement (BoardView).
export const seatEdge = (nPlayers: number, seat: number): Edge =>
  (nPlayers === 2 ? EDGES_2 : EDGES_4)[seat] ?? "bottom";

export default function PlayerAreas({
  board,
  oceanX,
  oceanY,
  oceanW,
  oceanH,
  band,
  cardW,
  cardH,
  hex,
  partners,
  onPartner,
}: {
  board: Board;
  // The ocean rectangle; areas sit in the table band of width `band` around it.
  oceanX: number;
  oceanY: number;
  oceanW: number;
  oceanH: number;
  band: number;
  cardW: number;
  cardH: number;
  hex: number;
  // Seats the viewer may propose a trade to, and the click handler.
  partners?: Set<number>;
  onPartner?: (p: number, el: SVGGraphicsElement) => void;
}) {
  const edges = board.players.length === 2 ? EDGES_2 : EDGES_4;
  const centre = (edge: Edge): { x: number; y: number } => {
    switch (edge) {
      case "bottom":
        return { x: oceanX + oceanW / 2, y: oceanY + oceanH + band / 2 };
      case "top":
        return { x: oceanX + oceanW / 2, y: oceanY - band / 2 };
      case "left":
        return { x: oceanX - band / 2, y: oceanY + oceanH / 2 };
      case "right":
        return { x: oceanX + oceanW + band / 2, y: oceanY + oceanH / 2 };
    }
  };

  return (
    <g>
      {board.players.map((p, i) => {
        const edge = edges[i] ?? "bottom";
        const { x, y } = centre(edge);
        const builds = board.buildings.filter((b) => b.player === p.player);
        return (
          <g key={p.player} transform={`translate(${x} ${y}) rotate(${EDGE_ANGLE[edge]})`}>
            <Area
              player={p}
              roadsLeft={Math.max(0, ROAD_SUPPLY - board.roads.filter((r) => r.player === p.player).length)}
              settlementsLeft={Math.max(0, SETTLEMENT_SUPPLY - builds.filter((b) => b.kind === "settlement").length)}
              citiesLeft={Math.max(0, CITY_SUPPLY - builds.filter((b) => b.kind === "city").length)}
              cardW={cardW}
              cardH={cardH}
              hex={hex}
              onHand={partners?.has(p.player) ? (el) => onPartner?.(p.player, el) : undefined}
            />
          </g>
        );
      })}
    </g>
  );
}
