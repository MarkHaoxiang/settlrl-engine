import { useEffect, useMemo, useState } from "react";
import BoardView, { type BoardInteraction } from "../components/BoardView";
import ChatPanel from "../components/ChatPanel";
import NewGameDialog from "../components/NewGameDialog";
import TerrainIcon from "../components/TerrainIcon";
import TopBar from "../components/TopBar";
import { useGame } from "../lib/useGame";
import { actionMeta } from "../lib/actionMeta";
import type { GameAction } from "../lib/game";
import {
  PLAYER_COLORS,
  PLAYER_STROKES,
  TERRAIN_FILL,
  TERRAIN_STROKE,
  playerName,
  type DevCardKind,
  type Player,
  type ResourceKind,
} from "../lib/boardData";
import { cubeEq, edgeEq, hexEq, type Cube, type Hex } from "../lib/hex";
import {
  HIGHLIGHT,
  buttonStyle,
  overlayMsgStyle,
  panelStyle,
  selectedStyle,
} from "../lib/ui";

const RESOURCES: { key: ResourceKind; label: string }[] = [
  { key: "wood", label: "Wood" },
  { key: "brick", label: "Brick" },
  { key: "sheep", label: "Sheep" },
  { key: "wheat", label: "Wheat" },
  { key: "ore", label: "Ore" },
];

const DEV_CARDS: { key: DevCardKind; label: string; icon: string }[] = [
  { key: "knight", label: "Knight", icon: "⚔️" },
  { key: "road_building", label: "Road building", icon: "🚧" },
  { key: "year_of_plenty", label: "Year of plenty", icon: "🎁" },
  { key: "monopoly", label: "Monopoly", icon: "🎩" },
  { key: "victory_point", label: "Victory point", icon: "⭐" },
];

const DEV_FILL = "#5B4B8A";
const DEV_STROKE = "#3C3160";

// Action types that are placed by clicking the board (vs. fired by a button or
// chosen from a resource popover).
const BOARD_TYPES = new Set([
  "setup_settlement",
  "build_settlement",
  "build_city",
  "setup_road",
  "build_road",
  "move_robber",
  "play_knight",
]);
// Action types whose concrete variants are chosen from a popover list.
const RESOURCE_TYPES = new Set(["play_monopoly", "play_year_of_plenty", "maritime_trade"]);

const PHASE_LABEL: Record<string, string> = {
  setup_settlement: "Setup",
  setup_road: "Setup",
  roll: "Roll",
  discard: "Discard",
  move_robber: "Robber",
  main: "Main",
  game_over: "Game over",
};

const smallButton: React.CSSProperties = { ...buttonStyle, padding: "5px 12px", fontSize: 12 };

// A hand chip: the count over a faded background icon (the card's name is the
// hover tooltip).
function Chip({
  count,
  label,
  icon,
  fill,
  stroke,
}: {
  count: number;
  label: string;
  icon: React.ReactNode;
  fill: string;
  stroke: string;
}) {
  const empty = count === 0;
  return (
    <div
      title={label}
      style={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 40,
        height: 34,
        borderRadius: 8,
        background: fill,
        border: `2px solid ${stroke}`,
        opacity: empty ? 0.4 : 1,
      }}
    >
      <span
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {icon}
      </span>
      <span
        style={{
          position: "relative",
          fontSize: 16,
          fontWeight: 800,
          lineHeight: 1,
          color: "#1a1a1a",
          // A halo in the chip colour keeps the digit legible over the icon.
          textShadow: `0 0 4px ${fill}, 0 0 4px ${fill}, 0 0 3px ${fill}`,
        }}
      >
        {count}
      </span>
    </div>
  );
}

// The acting human's hand: resources + dev cards by type, on a single row.
function Hand({ player, you }: { player: Player; you: boolean }) {
  const color = PLAYER_COLORS[player.player] ?? "#888";
  const stroke = PLAYER_STROKES[player.player] ?? "#444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginRight: 8 }}>
        <span style={{ width: 14, height: 14, borderRadius: "50%", background: color, border: `2px solid ${stroke}` }} />
        <span style={{ fontWeight: 700, fontSize: 13 }}>
          {playerName(player.player)}
          {you ? " (you)" : ""}
        </span>
      </div>
      {RESOURCES.map((r) => (
        <Chip
          key={r.key}
          count={player.resources[r.key]}
          label={r.label}
          // The board tiles' motif, so chips match the terrain they come from.
          icon={
            <svg width={28} height={28} viewBox="-11 -11 22 22">
              <TerrainIcon terrain={r.key} cx={0} cy={0} scale={1.1} opacity={0.9} />
            </svg>
          }
          fill={TERRAIN_FILL[r.key]}
          stroke={TERRAIN_STROKE[r.key]}
        />
      ))}
      <span style={{ width: 1, alignSelf: "stretch", background: "rgba(255,255,255,0.15)", margin: "0 8px" }} />
      {DEV_CARDS.map((d) => (
        <Chip
          key={d.key}
          count={player.devCardTypes[d.key]}
          label={d.label}
          icon={<span style={{ fontSize: 17, opacity: 0.8 }}>{d.icon}</span>}
          fill={DEV_FILL}
          stroke={DEV_STROKE}
        />
      ))}
    </div>
  );
}

// A popover listing concrete actions to pick from (resource trades, or the
// victim choice when a robber tile has several stealable players).
function ChoicePopover({ actions, onPick, onClose }: { actions: GameAction[]; onPick: (flat: number) => void; onClose: () => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, borderTop: "1px solid rgba(255,255,255,0.12)", paddingTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 }}>Choose</span>
        <button style={{ ...smallButton, padding: "2px 10px" }} onClick={onClose}>
          Cancel
        </button>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", maxWidth: 640 }}>
        {actions.map((a) => (
          <button key={a.flat} style={buttonStyle} onClick={() => onPick(a.flat)}>
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function PlayView() {
  const { snapshot, error, busy, act, reset, chat } = useGame();
  const [armed, setArmed] = useState<string | null>(null);
  const [choice, setChoice] = useState<GameAction[] | null>(null);
  // Whether the new-game configuration dialog is open (shown on entry, and
  // reopened by the New game button).
  const [configuring, setConfiguring] = useState(true);

  const actions = snapshot?.actions ?? [];

  // Reset transient UI when a new snapshot arrives: drop any popover, keep the
  // armed type only if it's still available, and auto-arm setup placements.
  useEffect(() => {
    if (!snapshot) return;
    setChoice(null);
    const types = new Set(snapshot.actions.map((a) => a.type));
    setArmed((prev) => {
      if (prev && types.has(prev)) return prev;
      if (types.has("setup_settlement")) return "setup_settlement";
      if (types.has("setup_road")) return "setup_road";
      return null;
    });
  }, [snapshot]);

  // Esc disarms / closes the popover.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setArmed(null);
        setChoice(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const byType = (type: string) => actions.filter((a) => a.type === type);
  const availableTypes = useMemo(() => Array.from(new Set(actions.map((a) => a.type))), [actions]);

  // Either apply the single matching action, or pop a chooser for the rest.
  const actOrChoose = (matches: GameAction[]) => {
    if (matches.length === 1) act(matches[0].flat);
    else if (matches.length > 1) setChoice(matches);
  };

  // Board click targets for the armed action type.
  const interaction: BoardInteraction | undefined = useMemo(() => {
    if (!armed || !BOARD_TYPES.has(armed)) return undefined;
    const armedActions = byType(armed);
    const tiles: Hex[] = [];
    for (const a of armedActions) if (a.tile && !tiles.some((t) => hexEq(t, a.tile!))) tiles.push(a.tile);
    return {
      vertices: armedActions.filter((a) => a.vertex).map((a) => a.vertex as Cube),
      edges: armedActions.filter((a) => a.edge).map((a) => a.edge as { a: Cube; b: Cube }),
      tiles,
      onVertex: (v) => actOrChoose(armedActions.filter((a) => a.vertex && cubeEq(a.vertex, v))),
      onEdge: (e) => actOrChoose(armedActions.filter((a) => a.edge && edgeEq(a.edge, e))),
      onTile: (t) => actOrChoose(armedActions.filter((a) => a.tile && hexEq(a.tile, t))),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [armed, actions]);

  if (error) return <div style={overlayMsgStyle}>{error}</div>;
  if (!snapshot) return <div style={overlayMsgStyle}>Loading game…</div>;

  const { status, board } = snapshot;
  // Hotseat: the hand panel follows whichever human seat is acting (falling
  // back to the first human while bots play out or at game over). With no
  // human seats at all the game is a spectated bot match: no hand panel.
  const seats = status.seats;
  const soloHuman = seats.filter((s) => s === "human").length === 1;
  const handSeat = seats[status.acting_player] === "human" ? status.acting_player : seats.indexOf("human");
  const me = handSeat >= 0 ? board.players[handSeat] : null;
  const winnerLabel =
    status.winner == null
      ? ""
      : seats[status.winner] === "human"
        ? soloHuman
          ? "You win! 🎉"
          : `${playerName(status.winner)} wins! 🎉`
        : `${playerName(status.winner)} wins`;

  // The action buttons, derived from the distinct legal action types.
  const onTypeButton = (type: string) => {
    if (BOARD_TYPES.has(type)) setArmed((prev) => (prev === type ? null : type));
    else if (RESOURCE_TYPES.has(type)) setChoice(byType(type));
    else act(byType(type)[0].flat); // parameterless: a single flat action
  };

  return (
    <div style={{ display: "flex", width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Board area: the chrome inside is anchored to it, not the viewport */}
      <div style={{ position: "relative", flex: 1, overflow: "hidden" }}>
        <BoardView board={board} interaction={interaction} />
        <TopBar mode="Play" />

        {status.terminal && (
          <div
            style={{
              ...panelStyle,
              position: "absolute",
              top: 70,
              left: "50%",
              transform: "translateX(-50%)",
              padding: "10px 20px",
              color: HIGHLIGHT,
              fontWeight: 700,
              zIndex: 10,
            }}
          >
            {winnerLabel}
          </div>
        )}

        {/* Full-width flex strip so the panel centres without halving the
            shrink-to-fit width (which would wrap the hand). */}
        <div
          style={{
            position: "absolute",
            bottom: 16,
            left: 0,
            right: 0,
            display: "flex",
            justifyContent: "center",
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              ...panelStyle,
              pointerEvents: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              padding: "10px 16px",
              borderRadius: 16,
              boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
              maxWidth: "94%",
            }}
          >
            {me && <Hand player={me} you={soloHuman} />}

            {/* Status + action buttons share one row */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                fontSize: 13,
                flexWrap: "wrap",
                ...(me ? { borderTop: "1px solid rgba(255,255,255,0.12)", paddingTop: 8 } : {}),
              }}
            >
              <span style={{ fontWeight: 700, opacity: 0.85 }}>{PHASE_LABEL[status.phase] ?? status.phase}</span>
              {status.dice_roll > 0 && <span style={{ opacity: 0.85 }}>🎲 {status.dice_roll}</span>}
              {snapshot.bot_move && (
                <span
                  className="fade-in"
                  title={`${playerName(snapshot.bot_move.player)} · ${snapshot.bot_move.action.label}`}
                  style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                >
                  <span
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: PLAYER_COLORS[snapshot.bot_move.player] ?? "#888",
                    }}
                  />
                  <span style={{ fontSize: 16 }}>{actionMeta(snapshot.bot_move.action.type).icon}</span>
                </span>
              )}
              <span style={{ opacity: 0.6 }}>
                {status.terminal
                  ? "Game over"
                  : status.your_turn
                    ? soloHuman
                      ? "Your turn"
                      : `${playerName(status.acting_player)}'s turn`
                    : `${playerName(status.acting_player)} is thinking…`}
              </span>
              {!status.terminal &&
                status.your_turn &&
                availableTypes.map((type) => (
                  <button
                    key={type}
                    title={actionMeta(type).label}
                    style={{
                      ...buttonStyle,
                      fontSize: 18,
                      padding: "5px 12px",
                      lineHeight: 1.2,
                      ...(armed === type ? selectedStyle : {}),
                    }}
                    disabled={busy}
                    onClick={() => onTypeButton(type)}
                  >
                    {actionMeta(type).icon}
                  </button>
                ))}
              <button style={{ ...smallButton, marginLeft: "auto" }} onClick={() => setConfiguring(true)}>
                New game
              </button>
            </div>

            {choice && (
              <ChoicePopover
                actions={choice}
                onPick={(flat) => {
                  setChoice(null);
                  act(flat);
                }}
                onClose={() => setChoice(null)}
              />
            )}
          </div>
        </div>
      </div>

      <ChatPanel
        entries={snapshot.log}
        onSend={(text) => chat(text, handSeat >= 0 ? handSeat : null)}
      />

      {configuring && (
        <NewGameDialog
          onStart={(config) => {
            setConfiguring(false);
            reset(config);
          }}
          onClose={() => setConfiguring(false)}
        />
      )}
    </div>
  );
}
