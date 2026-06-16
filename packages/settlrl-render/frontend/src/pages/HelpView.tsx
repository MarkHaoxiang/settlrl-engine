// Help page: how to drive a game through this UI. The explanatory text that
// used to crowd the Play control bar lives here instead.

import { Fragment } from "react";
import { Link } from "react-router-dom";
import { ACTION_META } from "../lib/actionMeta";
import { LINK, panelStyle } from "../lib/ui";

// What each control does, keyed by action type and grouped by where it lives:
// directly on the board, on a hand chip, or on the bottom bar (display order).
const BOARD_HELP: [string, string][] = [
  ["roll_dice", "The dice rest by the board's corner — click them when they glow to roll."],
  ["build_settlement", "A dot on a corner: click it to build a settlement there."],
  ["build_city", "A larger dot on your settlement: click it to upgrade to a city."],
  ["build_road", "A dash on an edge: click it to build a road there."],
  ["move_robber", "A 7 was rolled: click a highlighted tile, then pick who to rob."],
  ["maritime_trade", "Click the bank pile you want: trade at 4:1, or better via your ports."],
  ["propose_trade", "Click an opponent's hand pile: a two-sided composer shows your counts and what card counting proves about theirs — pick a card from each side and propose."],
];
const HAND_HELP: [string, string][] = [
  ["play_knight", "Click the card, then a tile for the robber (and who to rob)."],
  ["play_road_building", "Click the card, then place two free roads on the board."],
  ["play_monopoly", "Click the card and pick a resource to take from everyone."],
  ["play_year_of_plenty", "Click the card and pick two resources from the bank."],
  ["discard", "A 7 costs you half your hand: click resource cards to discard them."],
];
const BAR_HELP: [string, string][] = [
  ["buy_development_card", "Buy a development card."],
  ["accept_trade", "Take a trade you've been offered."],
  ["reject_trade", "Turn a trade down."],
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
            Scroll or pinch to zoom, drag to pan, and spin the table a quarter turn with
            the ↺ ↻ buttons (bottom-left). All of it works from the keyboard too: arrow keys
            pan, + / − zoom, [ and ] spin, and 0 re-fits the whole table in view. Every build you can afford right now is
            marked on the board in your colour — a small dot on a corner, a short dash on an
            edge. Hover one to preview the piece, then click and confirm in the popup, which
            lists the build cost. Esc or a click elsewhere cancels. The scene is the whole
            table, seen from above: the bank's card piles sit to the left (counts on the
            piles), and each seat's play area lines its table edge — their face-down hand
            and dev cards plus every road, settlement, and city still in their box.
          </span>
          <ActionTable rows={BOARD_HELP} />
        </Section>

        <Section title="Your hand">
          <span style={{ opacity: 0.8 }}>
            The chips in the bottom panel are the acting human's hand: resources and development
            cards by type. Glowing chips are playable — click one:
          </span>
          <ActionTable rows={HAND_HELP} />
        </Section>

        <Section title="The bar">
          <span style={{ opacity: 0.8 }}>
            The bottom bar keeps the turn-flow moves, one button per move currently available:
          </span>
          <ActionTable rows={BAR_HELP} />
        </Section>

        <Section title="The top bar">
          <span>
            Back to the menu, the light/dark theme toggle, and in Play: <b>New game</b>
            (reconfigure seats; cancelling keeps the game in progress).
          </span>
        </Section>

        <Section title="Players & chat">
          <span>
            The right column opens with the seats in playing order — ⭐ victory points,
            🎴 resource cards, 🃏 development cards, the acting seat tinted — over the
            chat / game log. <b>🔍 Inspect</b> on an opponent unfolds card counting: proven
            per-resource bounds on their hand, where "2" is a known count and "0–3" is what
            a hidden robber steal allows. Only public information is used, so it never
            reveals anything you couldn't have tracked yourself.
          </span>
        </Section>

        <Section title="Seats & turns">
          <span>
            Each seat is a human or a bot, chosen in the New game dialog. Creating a game claims
            every human seat on this screen (hotseat); the 🔗 button copies an invite link, and
            opening it claims a free human seat instead — or spectates when none is left. Hands
            are private on the server: you only ever receive your own cards. While bots play, the
            bar shows who's thinking and a chip with each move as it lands; with no human seats
            the game simply plays itself. Some moves are forced out of turn — a 7 can make
            everyone discard — so watch the status line.
          </span>
        </Section>

        <Section title="Replays">
          <span>
            The Replay view steps through a recorded game. Load a saved record file (💾 saves
            one) or pull in your last finished game (records only export once a game is over —
            replaying one reconstructs hidden hands),
            then scrub with the slider, step move by move, or press play — the log fills in as the
            game advances.
          </span>
        </Section>
      </div>
    </div>
  );
}
