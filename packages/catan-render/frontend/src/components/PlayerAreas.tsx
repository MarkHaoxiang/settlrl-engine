import { PLAYER_COLORS, PLAYER_STROKES, type Board, type Player } from "../lib/boardData";
import { housePath } from "./Building";

// Card backs: the resource hand (leather tan) and development cards (purple).
const HAND_BACK = "#C9A66B";
const HAND_EDGE = "#7A5C33";
const DEV_BACK = "#5B4B8A";
const DEV_EDGE = "#3C3160";

// Piece supply per player (base game).
const ROAD_SUPPLY = 15;
const SETTLEMENT_SUPPLY = 5;
const CITY_SUPPLY = 4;

// A face-down pile of cards with a count token (nothing is drawn when empty —
// an empty hand leaves bare table).
function FaceDownPile({
  cx,
  cy,
  w,
  h,
  count,
  fill,
  edge,
  owner,
  label,
}: {
  cx: number;
  cy: number;
  w: number;
  h: number;
  count: number;
  fill: string;
  edge: string;
  // Player colour, striped along the pile's top edge so piles stay owned.
  owner: string;
  label: string;
}) {
  if (count === 0) return null;
  const card = { x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: w * 0.09 };
  return (
    <g>
      <title>{`${label}: ${count}`}</title>
      {count > 2 && (
        <rect {...card} transform={`rotate(-3 ${cx} ${cy})`} fill={fill} stroke={edge} strokeWidth={1.5} opacity={0.85} />
      )}
      <rect {...card} fill={fill} stroke={edge} strokeWidth={1.5} />
      <rect
        x={cx - w * 0.32}
        y={cy - h * 0.32}
        width={w * 0.64}
        height={h * 0.64}
        rx={w * 0.06}
        fill="none"
        stroke={edge}
        strokeWidth={1.2}
        opacity={0.6}
      />
      <rect x={card.x} y={card.y} width={card.width} height={h * 0.08} rx={w * 0.04} fill={owner} opacity={0.9} />
      <circle cx={cx} cy={cy + h * 0.26} r={h * 0.13} fill="#FDF6E3" stroke="#A08050" strokeWidth={1.2} />
      <text
        x={cx}
        y={cy + h * 0.26}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={h * 0.14}
        fontWeight="bold"
        fill="#2C1A00"
        fontFamily="Georgia, 'Times New Roman', serif"
      >
        {count}
      </text>
    </g>
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
}: {
  player: Player;
  roadsLeft: number;
  settlementsLeft: number;
  citiesLeft: number;
  cardW: number;
  cardH: number;
  hex: number;
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
  const total = pilesW + 40 + piecesW;
  let x = -total / 2;

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
      <FaceDownPile
        cx={handX}
        cy={0}
        w={cardW}
        h={cardH}
        count={player.resourceCards}
        fill={HAND_BACK}
        edge={HAND_EDGE}
        owner={color}
        label={`${"resource cards"}`}
      />
      <FaceDownPile
        cx={devX}
        cy={0}
        w={cardW}
        h={cardH}
        count={player.devCards}
        fill={DEV_BACK}
        edge={DEV_EDGE}
        owner={color}
        label="development cards"
      />
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
type Edge = (typeof EDGES_4)[number];

const EDGE_ANGLE: Record<Edge, number> = { bottom: 0, left: 90, top: 180, right: -90 };

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
        const roadsLeft = ROAD_SUPPLY - board.roads.filter((r) => r.player === p.player).length;
        const builds = board.buildings.filter((b) => b.player === p.player);
        const settlementsLeft =
          SETTLEMENT_SUPPLY - builds.filter((b) => b.kind === "settlement").length;
        const citiesLeft = CITY_SUPPLY - builds.filter((b) => b.kind === "city").length;
        return (
          <g key={p.player} transform={`translate(${x} ${y}) rotate(${EDGE_ANGLE[edge]})`}>
            <Area
              player={p}
              roadsLeft={Math.max(0, roadsLeft)}
              settlementsLeft={Math.max(0, settlementsLeft)}
              citiesLeft={Math.max(0, citiesLeft)}
              cardW={cardW}
              cardH={cardH}
              hex={hex}
            />
          </g>
        );
      })}
    </g>
  );
}
