// Help page: how to drive a game through this UI. The explanatory text that
// used to crowd the Play control bar lives here instead.

import { Fragment } from "react";
import { Link } from "react-router-dom";
import { ACTION_META } from "../lib/actionMeta";
import { LINK, panelStyle } from "../lib/ui";

// What each control does, keyed by action type and grouped by where it lives:
// directly on the board, on a hand chip, or on the bottom bar (display order).
const BOARD_HELP: [string, string][] = [
  ["build_settlement", "A faint house on a corner: click it to build a settlement there."],
  ["build_city", "A dashed outline over your settlement: click it to upgrade to a city."],
  ["build_road", "A faint dashed edge: click it to build a road there."],
  ["move_robber", "A 7 was rolled: click a highlighted tile, then pick who to rob."],
];
const HAND_HELP: [string, string][] = [
  ["play_knight", "Click the card, then a tile for the robber (and who to rob)."],
  ["play_road_building", "Click the card, then place two free roads on the board."],
  ["play_monopoly", "Click the card and pick a resource to take from everyone."],
  ["play_year_of_plenty", "Click the card and pick two resources from the bank."],
  ["discard", "A 7 costs you half your hand: click resource cards to discard them."],
];
const BAR_HELP: [string, string][] = [
  ["roll_dice", "Roll the dice to start your turn."],
  ["buy_development_card", "Buy a development card."],
  ["maritime_trade", "Trade with the bank (4:1, or better via your ports)."],
  ["end_turn", "End your turn."],
];

function ActionTable({ rows }: { rows: [string, string][] }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto auto 1fr", gap: "6px 12px", alignItems: "baseline" }}>
      {rows.map(([type, text]) => (
        <Fragment key={type}>
          <span style={{ fontSize: 18 }}>{ACTION_META[type].icon}</span>
          <span style={{ fontWeight: 700, whiteSpace: "nowrap" }}>{ACTION_META[type].label}</span>
          <span style={{ opacity: 0.8 }}>{text}</span>
        </Fragment>
      ))}
    </div>
  );
}

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
          <Link to="/" style={{ color: LINK, fontSize: 13, marginLeft: "auto" }}>
            ← Menu
          </Link>
        </div>

        <Section title="The board">
          <span>
            Scroll or pinch to zoom, and drag to pan. Everything you can build right now is
            ghosted on the board in your colour (so the ghosts also show what you can afford) —
            click one and confirm in the popup, which lists the build cost. Esc or a click
            elsewhere cancels.
          </span>
          <ActionTable rows={BOARD_HELP} />
        </Section>

        <Section title="Your hand">
          <span style={{ opacity: 0.8 }}>
            The chips in the bottom panel are the acting human's hand: resources and development
            cards by type (the corner panels track every player's card counts and victory
            points). Glowing chips are playable — click one:
          </span>
          <ActionTable rows={HAND_HELP} />
        </Section>

        <Section title="The bar">
          <span style={{ opacity: 0.8 }}>
            The bottom bar keeps the turn-flow moves, one button per move currently available:
          </span>
          <ActionTable rows={BAR_HELP} />
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

        <Section title="Replays">
          <span>
            The Replay view steps through a recorded game. Load a saved record file (💾 saves one;
            so does <code>GET /api/game/record</code>) or pull in the live game as played so far,
            then scrub with the slider, step move by move, or press play — the log fills in as the
            game advances.
          </span>
        </Section>
      </div>
    </div>
  );
}
