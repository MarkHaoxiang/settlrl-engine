// Icon + short label per action type, shared by the Play control bar (icon
// buttons with the label as tooltip) and the help page (which documents what
// each icon does).

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
  discard: { icon: "🗑️", label: "Discard" },
};

export const actionMeta = (type: string): ActionMeta =>
  ACTION_META[type] ?? { icon: "❔", label: type };
