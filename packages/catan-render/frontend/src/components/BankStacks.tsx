import {
  DEV_CARD_BACK,
  RESOURCE_ORDER,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  type Bank,
  type ResourceKind,
} from "../lib/boardData";
import { HIGHLIGHT } from "../lib/ui";
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
  tradable,
  onPick,
}: {
  bank: Bank;
  // Grid centre.
  cx: number;
  cy: number;
  cardW: number;
  cardH: number;
  // Piles the viewer can currently trade for (bank rates); they highlight and
  // take a click, reported with the clicked element for popover anchoring.
  tradable?: Set<ResourceKind>;
  onPick?: (r: ResourceKind, el: SVGGraphicsElement) => void;
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
        const clickable = tradable?.has(r) ?? false;
        const pile = (
          <CardPile
            cx={x}
            cy={y}
            w={cardW}
            h={cardH}
            count={bank.resources[r]}
            fill={TERRAIN_FILL[r]}
            stroke={TERRAIN_STROKE[r]}
            label={clickable ? `${r} left — click to trade for one` : `${r} left`}
          >
            <TerrainIcon terrain={r} cx={x} cy={y - cardH * 0.16} scale={cardW * 0.016} opacity={0.85} />
          </CardPile>
        );
        // data-bank tags each pile as a fly-token endpoint (TransferAnimations).
        if (!clickable) return <g key={r} data-bank={r}>{pile}</g>;
        return (
          <g key={r} data-bank={r} className="board-ghost" onClick={(e) => onPick?.(r, e.currentTarget)}>
            <rect
              className="ghost"
              x={x - cardW / 2 - 6}
              y={y - cardH / 2 - 6}
              width={cardW + 12}
              height={cardH + 12}
              rx={10}
              fill="none"
              stroke={HIGHLIGHT}
              strokeWidth={3}
            />
            {pile}
          </g>
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
