// Flies a chip across the table for each card transfer (lib/transfers). Tokens
// are positioned by measuring their endpoint elements (the bank piles and each
// seat's resource hand pile, tagged with data-bank / data-seat) at fire time,
// so they track the live pan / zoom / rotation without re-deriving the board
// geometry. Built
// imperatively (Web Animations API) — the chips are transient and never re-read
// by React, so they don't belong in the render tree.

import { useEffect, useRef, type RefObject } from "react";
import { HAND_CARD_BACK, TERRAIN_FILL, TERRAIN_STROKE } from "../lib/boardData";
import type { Anchor, FlyToken } from "../lib/transfers";

const CHIP_W = 20;
const CHIP_H = 28;
const FLY_MS = 620;
const STAGGER_MS = 70;

function chipStyle(resource: FlyToken["resource"]): Partial<CSSStyleDeclaration> {
  const back = resource ? { fill: TERRAIN_FILL[resource], stroke: TERRAIN_STROKE[resource] } : HAND_CARD_BACK;
  return {
    position: "absolute",
    left: "0",
    top: "0",
    width: `${CHIP_W}px`,
    height: `${CHIP_H}px`,
    borderRadius: "4px",
    background: back.fill,
    border: `1.5px solid ${back.stroke}`,
    boxShadow: "0 3px 8px rgba(0,0,0,0.4)",
    willChange: "transform, opacity",
  };
}

export default function TransferAnimations({
  tokens,
  containerRef,
}: {
  tokens: FlyToken[];
  containerRef: RefObject<HTMLDivElement | null>;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  // Ids already animated, so a re-run with the same batch (e.g. dev-mode double
  // effects) doesn't fire a chip twice.
  const fired = useRef<Set<string>>(new Set());

  useEffect(() => {
    const layer = layerRef.current;
    const container = containerRef.current;
    if (!layer || !container || tokens.length === 0) return;
    const c = container.getBoundingClientRect();
    const locate = (a: Anchor): { x: number; y: number } | null => {
      const sel =
        a.kind === "seat"
          ? `[data-seat="${a.seat}"]`
          : a.resource
            ? `[data-bank="${a.resource}"]`
            : "[data-bank]";
      const el = container.querySelector(sel);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      // A hand pile drawn empty (count 0) collapses to a zero box at the SVG
      // origin — skip it rather than fly a chip from the corner.
      if (r.width === 0 && r.height === 0) return null;
      return { x: r.x + r.width / 2 - c.x - CHIP_W / 2, y: r.y + r.height / 2 - c.y - CHIP_H / 2 };
    };

    let flown = 0;
    for (const t of tokens) {
      if (fired.current.has(t.id)) continue;
      fired.current.add(t.id);
      const from = locate(t.from);
      const to = locate(t.to);
      if (!from || !to) continue;
      const chip = document.createElement("div");
      Object.assign(chip.style, chipStyle(t.resource));
      layer.appendChild(chip);
      const anim = chip.animate(
        [
          { transform: `translate(${from.x}px, ${from.y}px) scale(0.5)`, opacity: 0 },
          { transform: `translate(${from.x}px, ${from.y}px) scale(1)`, opacity: 1, offset: 0.18 },
          { transform: `translate(${to.x}px, ${to.y}px) scale(1)`, opacity: 1, offset: 0.85 },
          { transform: `translate(${to.x}px, ${to.y}px) scale(0.55)`, opacity: 0 },
        ],
        { duration: FLY_MS, delay: flown * STAGGER_MS, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "both" }
      );
      anim.onfinish = () => chip.remove();
      flown++;
    }
    // Keep the dedupe set from growing without bound across a long game.
    if (fired.current.size > 256) fired.current.clear();
  }, [tokens, containerRef]);

  return <div ref={layerRef} className="transfer-layer" style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 8 }} />;
}
