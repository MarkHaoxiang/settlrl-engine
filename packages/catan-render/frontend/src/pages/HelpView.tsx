// Help page: how to drive a game through this UI. The explanatory text that
// used to crowd the Play control bar lives here instead.

import { Fragment } from "react";
import { Link } from "react-router-dom";
import { ACTION_META } from "../lib/actionMeta";
import { panelStyle } from "../lib/ui";

// What each control-bar icon does, keyed by action type (display order).
const ACTION_HELP: [string, string][] = [
  ["roll_dice", "Roll the dice to start your turn."],
  ["build_settlement", "Build a settlement — click a highlighted vertex."],
  ["build_city", "Upgrade a settlement to a city — click a highlighted vertex."],
  ["build_road", "Build a road — click a highlighted edge."],
  ["buy_development_card", "Buy a development card."],
  ["play_knight", "Play a knight: move the robber — click a highlighted tile."],
  ["move_robber", "A 7 was rolled: move the robber — click a highlighted tile."],
  ["play_road_building", "Play Road Building: place two free roads."],
  ["play_monopoly", "Play Monopoly: pick a resource to take from everyone."],
  ["play_year_of_plenty", "Play Year of Plenty: pick two resources from the bank."],
  ["maritime_trade", "Trade with the bank (4:1, or better via your ports)."],
  ["discard", "Half your hand is lost to a 7: pick the cards to discard."],
  ["end_turn", "End your turn."],
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <span style={{ fontSize: 12, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 }}>
        {title}
      </span>
      {children}
    </div>
  );
}

export default function HelpView() {
  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        padding: "40px 16px",
        overflowY: "auto",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          ...panelStyle,
          display: "flex",
          flexDirection: "column",
          gap: 20,
          padding: "24px 28px",
          maxWidth: 620,
          height: "fit-content",
          fontSize: 14,
          lineHeight: 1.5,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
          <span style={{ fontSize: 20, fontWeight: 700 }}>How to play</span>
          <Link to="/" style={{ color: "#9ec5e8", fontSize: 13, marginLeft: "auto" }}>
            ← Menu
          </Link>
        </div>

        <Section title="The board">
          <span>
            Scroll or pinch to zoom, and drag to pan. When a move targets the board, the legal
            spots light up — click a highlighted vertex (settlement / city), edge (road), or
            tile (robber) to play there. Click the button again to put it away.
          </span>
        </Section>

        <Section title="Actions">
          <span style={{ opacity: 0.8 }}>
            The bar at the bottom offers one button per move you can currently make
            (hover one for its name):
          </span>
          <div style={{ display: "grid", gridTemplateColumns: "auto auto 1fr", gap: "6px 12px", alignItems: "baseline" }}>
            {ACTION_HELP.map(([type, text]) => (
              <Fragment key={type}>
                <span style={{ fontSize: 18 }}>{ACTION_META[type].icon}</span>
                <span style={{ fontWeight: 700, whiteSpace: "nowrap" }}>{ACTION_META[type].label}</span>
                <span style={{ opacity: 0.8 }}>{text}</span>
              </Fragment>
            ))}
          </div>
        </Section>

        <Section title="Seats & turns">
          <span>
            Each seat is a human or a bot, chosen in the New game dialog. With several humans the
            game is hotseat: the hand panel follows whichever human is acting. While bots play, the
            bar shows who's thinking and a chip with each move as it lands; with no human seats the
            game simply plays itself. Some moves are forced out of turn — a 7 can make everyone
            discard — so watch the status line.
          </span>
        </Section>

        <Section title="Your hand">
          <span>
            The bottom-left of the bar is the acting human's hand: resources and development cards
            by type. The corner panels track every player's card counts and victory points.
          </span>
        </Section>
      </div>
    </div>
  );
}
