import { useEffect, useMemo, useState } from "react";
import BoardView, { type BoardInteraction } from "../components/BoardView";
import NewGameDialog from "../components/NewGameDialog";
import TopBar from "../components/TopBar";
import { useGame } from "../lib/useGame";
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

const LOCAL_PLAYER = 0;

const RESOURCES: { key: ResourceKind; label: string }[] = [
  { key: "wood", label: "Wood" },
  { key: "brick", label: "Brick" },
  { key: "sheep", label: "Sheep" },
  { key: "wheat", label: "Wheat" },
  { key: "ore", label: "Ore" },
];

const DEV_CARDS: { key: DevCardKind; label: string }[] = [
  { key: "knight", label: "Knight" },
  { key: "road_building", label: "Roads" },
  { key: "year_of_plenty", label: "Plenty" },
  { key: "monopoly", label: "Mono" },
  { key: "victory_point", label: "VP" },
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

// Button label per action type (board / parameterless / resource group buttons).
const TYPE_LABEL: Record<string, string> = {
  setup_settlement: "Place settlement",
  build_settlement: "Settlement",
  build_city: "City",
  setup_road: "Place road",
  build_road: "Road",
  move_robber: "Move robber",
  play_knight: "Knight",
  roll_dice: "Roll dice",
  end_turn: "End turn",
  buy_development_card: "Buy dev card",
  play_road_building: "Road building",
  discard: "Discard",
  play_monopoly: "Monopoly",
  play_year_of_plenty: "Year of plenty",
  maritime_trade: "Trade",
};

const PHASE_LABEL: Record<string, string> = {
  setup_settlement: "Setup — place a settlement",
  setup_road: "Setup — place a road",
  roll: "Roll the dice",
  discard: "Discard cards",
  move_robber: "Move the robber",
  main: "Main phase",
  game_over: "Game over",
};

const smallButton: React.CSSProperties = { ...buttonStyle, padding: "5px 12px", fontSize: 12 };

function Chip({ count, label, fill, stroke }: { count: number; label: string; fill: string; stroke: string }) {
  const empty = count === 0;
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minWidth: 46,
        padding: "6px 8px",
        borderRadius: 10,
        background: fill,
        border: `2px solid ${stroke}`,
        opacity: empty ? 0.4 : 1,
      }}
    >
      <span style={{ fontSize: 20, fontWeight: 700, lineHeight: 1, color: "#1a1a1a" }}>{count}</span>
      <span style={{ fontSize: 10, marginTop: 2, color: "#1a1a1a", opacity: 0.8 }}>{label}</span>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10, opacity: 0.6, textTransform: "uppercase", letterSpacing: 1 }}>{title}</span>
      <div style={{ display: "flex", gap: 6 }}>{children}</div>
    </div>
  );
}

// The local player's hand: resources + dev cards by type.
function Hand({ player }: { player: Player }) {
  const color = PLAYER_COLORS[player.player] ?? "#888";
  const stroke = PLAYER_STROKES[player.player] ?? "#444";
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 20, flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, paddingBottom: 4 }}>
        <span style={{ width: 16, height: 16, borderRadius: "50%", background: color, border: `2px solid ${stroke}` }} />
        <span style={{ fontWeight: 700, fontSize: 14 }}>{playerName(player.player)} (you)</span>
      </div>
      <Group title="Resources">
        {RESOURCES.map((r) => (
          <Chip key={r.key} count={player.resources[r.key]} label={r.label} fill={TERRAIN_FILL[r.key]} stroke={TERRAIN_STROKE[r.key]} />
        ))}
      </Group>
      <Group title="Dev cards">
        {DEV_CARDS.map((d) => (
          <Chip key={d.key} count={player.devCardTypes[d.key]} label={d.label} fill={DEV_FILL} stroke={DEV_STROKE} />
        ))}
      </Group>
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
  const { snapshot, error, busy, act, reset } = useGame();
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
  const me = board.players[LOCAL_PLAYER];

  // The action buttons, derived from the distinct legal action types.
  const onTypeButton = (type: string) => {
    if (BOARD_TYPES.has(type)) setArmed((prev) => (prev === type ? null : type));
    else if (RESOURCE_TYPES.has(type)) setChoice(byType(type));
    else act(byType(type)[0].flat); // parameterless: a single flat action
  };

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden" }}>
      <BoardView board={board} interaction={interaction} />
      <TopBar mode="Play" />

      {configuring && (
        <NewGameDialog
          onStart={(config) => {
            setConfiguring(false);
            reset(config);
          }}
          onClose={() => setConfiguring(false)}
        />
      )}

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
          {status.winner === LOCAL_PLAYER ? "You win! 🎉" : `${playerName(status.winner ?? 0)} wins`}
        </div>
      )}

      <div style={{ position: "absolute", bottom: 20, left: "50%", transform: "translateX(-50%)" }}>
        <div
          style={{
            ...panelStyle,
            display: "flex",
            flexDirection: "column",
            gap: 12,
            padding: "14px 18px",
            borderRadius: 16,
            boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
            maxWidth: "92vw",
          }}
        >
          <Hand player={me} />

          {/* Status + actions */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10, borderTop: "1px solid rgba(255,255,255,0.12)", paddingTop: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14, fontSize: 13, opacity: 0.85 }}>
              <span style={{ fontWeight: 700 }}>{PHASE_LABEL[status.phase] ?? status.phase}</span>
              {status.dice_roll > 0 && <span>🎲 {status.dice_roll}</span>}
              <span style={{ opacity: 0.7 }}>
                {busy ? "Bots thinking…" : status.your_turn ? "Your turn" : status.terminal ? "Game over" : "Waiting…"}
              </span>
              {armed && BOARD_TYPES.has(armed) && !busy && (
                <span style={{ color: HIGHLIGHT }}>Click a highlighted spot to place ({TYPE_LABEL[armed]})</span>
              )}
              <button style={{ ...smallButton, marginLeft: "auto" }} onClick={() => setConfiguring(true)}>
                New game
              </button>
            </div>

            {!status.terminal && (
              <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
                {availableTypes.length === 0 && <span style={{ fontSize: 13, opacity: 0.6 }}>No moves available.</span>}
                {availableTypes.map((type) => (
                  <button
                    key={type}
                    style={{ ...buttonStyle, ...(armed === type ? selectedStyle : {}) }}
                    disabled={busy}
                    onClick={() => onTypeButton(type)}
                  >
                    {TYPE_LABEL[type] ?? type}
                  </button>
                ))}
              </div>
            )}

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
    </div>
  );
}
