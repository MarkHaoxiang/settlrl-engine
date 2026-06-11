// Action-type display metadata: icon + short label per type (control bar,
// help page, log), build prices, and the phrasing of board-popover buttons.

import { playerName, type ResourceKind } from "./boardData";
import type { GameAction } from "./game";

// Build prices (the rulebook's Building Costs card), one entry per card paid;
// shown beside build buttons as confirmation context.
export const BUILD_COSTS: Record<string, ResourceKind[]> = {
  build_road: ["brick", "wood"],
  build_settlement: ["brick", "wood", "sheep", "wheat"],
  build_city: ["ore", "ore", "ore", "wheat", "wheat"],
  buy_development_card: ["ore", "sheep", "wheat"],
};

export interface ActionMeta {
  icon: string;
  label: string;
}

export const ACTION_META: Record<string, ActionMeta> = {
  setup_settlement: { icon: "🏠", label: "Place settlement" },
  build_settlement: { icon: "🏠", label: "Build settlement" },
  build_city: { icon: "🏰", label: "Build city" },
  setup_road: { icon: "🛤️", label: "Place road" },
  build_road: { icon: "🛤️", label: "Build road" },
  roll_dice: { icon: "🎲", label: "Roll dice" },
  end_turn: { icon: "⏭️", label: "End turn" },
  buy_development_card: { icon: "🃏", label: "Buy development card" },
  play_knight: { icon: "⚔️", label: "Play knight" },
  move_robber: { icon: "🦹", label: "Move robber" },
  play_road_building: { icon: "🚧", label: "Road building" },
  play_monopoly: { icon: "🎩", label: "Monopoly" },
  play_year_of_plenty: { icon: "🎁", label: "Year of plenty" },
  maritime_trade: { icon: "🚢", label: "Maritime trade" },
  propose_trade: { icon: "🤝", label: "Propose trade" },
  accept_trade: { icon: "✅", label: "Accept trade" },
  reject_trade: { icon: "❌", label: "Reject trade" },
  discard: { icon: "🗑️", label: "Discard" },
};

export const actionMeta = (type: string): ActionMeta =>
  ACTION_META[type] ?? { icon: "❔", label: type };

const stealText = (victim: number | null | undefined) =>
  victim != null && victim >= 0 ? ` — steal from ${playerName(victim)}` : "";

// What a board-popover button offers, phrased as the move it confirms.
export function confirmLabel(a: GameAction): string {
  switch (a.type) {
    case "setup_settlement":
      return "Place settlement";
    case "build_settlement":
      return "Build settlement";
    case "build_city":
      return "Upgrade to city";
    case "setup_road":
      return "Place road";
    case "build_road":
      return "Build road";
    case "move_robber":
      return `Move robber${stealText(a.victim)}`;
    case "play_knight":
      return `Play knight${stealText(a.victim)}`;
    default:
      return a.label;
  }
}
