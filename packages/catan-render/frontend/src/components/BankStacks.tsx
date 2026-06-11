import {
  DEV_CARD_BACK,
  RESOURCE_ORDER,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  type Bank,
} from "../lib/boardData";
import CardPile from "./CardPile";
import TerrainIcon from "./TerrainIcon";

// The bank as physical card piles on the table beside the board (cards at
// table scale): a two-column grid, one face-up pile per resource plus the
// development deck, drawn in board space so they pan and zoom with everything
// else.
export default function BankStacks({
  bank,
  cx,
  cy,
  cardW,
  cardH,
}: {
  bank: Bank;
  // Grid centre.
  cx: number;
  cy: number;
  cardW: number;
  cardH: number;
}) {
  const at = (i: number) => ({
    x: cx + ((i % 2) - 0.5) * (cardW + 18),
    y: cy + (Math.floor(i / 2) - 1) * (cardH + 20),
  });
  const dev = at(RESOURCE_ORDER.length);
  return (
    <g>
      {RESOURCE_ORDER.map((r, i) => {
        const { x, y } = at(i);
        return (
          <CardPile
            key={r}
            cx={x}
            cy={y}
            w={cardW}
            h={cardH}
            count={bank.resources[r]}
            fill={TERRAIN_FILL[r]}
            stroke={TERRAIN_STROKE[r]}
            label={`${r} left`}
          >
            <TerrainIcon terrain={r} cx={x} cy={y - cardH * 0.16} scale={cardW * 0.016} opacity={0.85} />
          </CardPile>
        );
      })}
      <CardPile
        cx={dev.x}
        cy={dev.y}
        w={cardW}
        h={cardH}
        count={bank.devCards}
        fill={DEV_CARD_BACK.fill}
        stroke={DEV_CARD_BACK.stroke}
        label="development cards left"
      >
        <path
          d={`M ${dev.x} ${dev.y - cardH * 0.3} l ${cardW * 0.14} ${cardH * 0.14} l ${-cardW * 0.14} ${cardH * 0.14} l ${-cardW * 0.14} ${-cardH * 0.14} z`}
          fill="none"
          stroke="#C9B7E8"
          strokeWidth={2.5}
          strokeLinejoin="round"
        />
      </CardPile>
    </g>
  );
}
