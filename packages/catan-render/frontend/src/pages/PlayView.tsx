import { useEffect, useMemo, useState } from "react";
import BoardView, { type BoardInteraction, type BoardTargetPoint } from "../components/BoardView";
import BoardPopover from "../components/BoardPopover";
import ChatPanel from "../components/ChatPanel";
import NewGameDialog from "../components/NewGameDialog";
import TerrainIcon from "../components/TerrainIcon";
import TopBar from "../components/TopBar";
import { useGame } from "../lib/useGame";
import { BUILD_COSTS, actionMeta } from "../lib/actionMeta";
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
import { cubeEq, edgeEq, hexEq, type Hex } from "../lib/hex";
import {
  ACCENT,
  ACCENT_GLOW,
  DIVIDER,
  buttonStyle,
  overlayMsgStyle,
  panelStyle,
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

// Board-targeted action types, by the target geometry they carry. These are
// always ghosted directly on the board — no arming step.
const VERTEX_KIND: Record<string, "settlement" | "city"> = {
  setup_settlement: "settlement",
  build_settlement: "settlement",
  build_city: "city",
};
const EDGE_TYPES = new Set(["setup_road", "build_road"]);

// The play action behind each dev-card hand chip (victory points are never
// played, so they have no entry).
const DEV_PLAY_TYPE: Partial<Record<DevCardKind, string>> = {
  knight: "play_knight",
  road_building: "play_road_building",
  year_of_plenty: "play_year_of_plenty",
  monopoly: "play_monopoly",
};

// Turn-flow actions that stay on the bottom bar (display order).
const BAR_TYPES = ["roll_dice", "buy_development_card", "maritime_trade", "end_turn"];

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

const stealText = (victim: number | null) =>
  victim != null && victim >= 0 ? ` — steal from ${playerName(victim)}` : "";

// What a board-popover button offers, phrased as the move it confirms.
function popupLabel(a: GameAction): string {
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

// A build price as a row of mini resource chips (board-popover buttons).
function CostRow({ cost }: { cost: ResourceKind[] }) {
  return (
    <span style={{ display: "inline-flex", gap: 2, marginLeft: 4 }}>
      {cost.map((r, i) => (
        <span
          key={i}
          title={r}
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            background: TERRAIN_FILL[r],
            border: `1px solid ${TERRAIN_STROKE[r]}`,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <svg width={12} height={12} viewBox="-11 -11 22 22">
            <TerrainIcon terrain={r} cx={0} cy={0} scale={0.9} />
          </svg>
        </span>
      ))}
    </span>
  );
}

// A hand chip: the count over a faded background icon (the card's name is the
// hover tooltip). Clickable chips glow; `selected` marks an armed card (the
// knight while choosing its robber tile).
function Chip({
  count,
  label,
  icon,
  fill,
  stroke,
  onClick,
  selected,
}: {
  count: number;
  label: string;
  icon: React.ReactNode;
  fill: string;
  stroke: string;
  onClick?: () => void;
  selected?: boolean;
}) {
  const empty = count === 0;
  return (
    <div
      title={label}
      onClick={onClick}
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
        ...(onClick ? { cursor: "pointer", boxShadow: ACCENT_GLOW } : {}),
        ...(selected ? { outline: `2px solid ${ACCENT}`, outlineOffset: 1 } : {}),
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
// Chips double as controls: dev cards in `playableDev` play on click, and
// resources in `discardable` discard one on click.
function Hand({
  player,
  you,
  discardable,
  onDiscard,
  playableDev,
  armedDev,
  onDev,
}: {
  player: Player;
  you: boolean;
  discardable?: Set<ResourceKind>;
  onDiscard?: (r: ResourceKind) => void;
  playableDev?: Set<DevCardKind>;
  armedDev?: DevCardKind | null;
  onDev?: (k: DevCardKind) => void;
}) {
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
      {RESOURCES.map((r) => {
        const canDiscard = discardable?.has(r.key) ?? false;
        return (
          <Chip
            key={r.key}
            count={player.resources[r.key]}
            label={canDiscard ? `${r.label} — click to discard one` : r.label}
            // The board tiles' motif, so chips match the terrain they come from.
            icon={
              <svg width={28} height={28} viewBox="-11 -11 22 22">
                <TerrainIcon terrain={r.key} cx={0} cy={0} scale={1.1} opacity={0.9} />
              </svg>
            }
            fill={TERRAIN_FILL[r.key]}
            stroke={TERRAIN_STROKE[r.key]}
            onClick={canDiscard ? () => onDiscard?.(r.key) : undefined}
          />
        );
      })}
      <span style={{ width: 1, alignSelf: "stretch", background: DIVIDER, margin: "0 8px" }} />
      {DEV_CARDS.map((d) => {
        const canPlay = playableDev?.has(d.key) ?? false;
        return (
          <Chip
            key={d.key}
            count={player.devCardTypes[d.key]}
            label={canPlay ? `${d.label} — click to play` : d.label}
            icon={<span style={{ fontSize: 17, opacity: 0.8 }}>{d.icon}</span>}
            fill={DEV_FILL}
            stroke={DEV_STROKE}
            onClick={canPlay ? () => onDev?.(d.key) : undefined}
            selected={armedDev === d.key}
          />
        );
      })}
    </div>
  );
}

// A popover listing concrete actions to pick from (resource choices: monopoly,
// year of plenty, maritime trade).
function ChoicePopover({ actions, onPick, onClose }: { actions: GameAction[]; onPick: (flat: number) => void; onClose: () => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, borderTop: `1px solid ${DIVIDER}`, paddingTop: 10 }}>
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
  // The chooser anchored to a clicked board target.
  const [popup, setPopup] = useState<{ actions: GameAction[]; x: number; y: number } | null>(null);
  // The bottom-panel resource chooser (monopoly / year of plenty / trade).
  const [choice, setChoice] = useState<GameAction[] | null>(null);
  // Knight targeting: set while the knight chip awaits its robber tile.
  const [knightArming, setKnightArming] = useState(false);
  // Whether the new-game configuration dialog is open (shown on entry, and
  // reopened by the New game button).
  const [configuring, setConfiguring] = useState(true);

  const actions = snapshot?.actions ?? [];

  // Reset transient UI when a new snapshot arrives.
  useEffect(() => {
    setPopup(null);
    setChoice(null);
    setKnightArming(false);
  }, [snapshot]);

  // Esc closes the choosers / cancels knight targeting.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPopup(null);
        setChoice(null);
        setKnightArming(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const byType = (type: string) => actions.filter((a) => a.type === type);
  const availableTypes = useMemo(() => new Set(actions.map((a) => a.type)), [actions]);

  // Ghost targets for every board-placeable action, all live at once. Knight
  // tiles only appear while the knight chip is armed — always-on they'd flood
  // the board any turn the card is in hand.
  const interaction: BoardInteraction | undefined = useMemo(() => {
    if (!snapshot || !snapshot.status.your_turn || snapshot.status.terminal) return undefined;
    const open = (list: GameAction[], at: BoardTargetPoint) => {
      if (list.length > 0) setPopup({ actions: list, x: at.x, y: at.y });
    };
    const verts = actions.filter((a) => a.vertex && VERTEX_KIND[a.type]);
    const edgeActs = actions.filter((a) => a.edge && EDGE_TYPES.has(a.type));
    const tileActs = byType("move_robber").concat(knightArming ? byType("play_knight") : []);
    if (verts.length === 0 && edgeActs.length === 0 && tileActs.length === 0) return undefined;
    const tiles: Hex[] = [];
    for (const a of tileActs) if (a.tile && !tiles.some((t) => hexEq(t, a.tile!))) tiles.push(a.tile);
    return {
      player: snapshot.status.acting_player,
      vertices: verts.map((a) => ({ cube: a.vertex!, kind: VERTEX_KIND[a.type] })),
      edges: edgeActs.map((a) => a.edge!),
      tiles,
      onVertex: (v, at) => open(verts.filter((a) => cubeEq(a.vertex!, v)), at),
      onEdge: (e, at) => open(edgeActs.filter((a) => edgeEq(a.edge!, e)), at),
      onTile: (t, at) => open(tileActs.filter((a) => a.tile && hexEq(a.tile, t)), at),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot, actions, knightArming]);

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

  // Hand-chip controls: only the acting human's own hand is live.
  const handActive = !status.terminal && status.your_turn && handSeat === status.acting_player && !busy;
  const discardActions = byType("discard");
  const discardable = handActive
    ? new Set(discardActions.map((a) => a.resource as ResourceKind))
    : undefined;
  const playableDev = handActive
    ? new Set(
        (Object.keys(DEV_PLAY_TYPE) as DevCardKind[]).filter((k) => availableTypes.has(DEV_PLAY_TYPE[k]!))
      )
    : undefined;

  const onDiscard = (r: ResourceKind) => {
    const m = discardActions.find((a) => a.resource === r);
    if (m) act(m.flat);
  };

  const onDev = (k: DevCardKind) => {
    if (k === "knight") {
      setKnightArming((v) => !v);
      setPopup(null);
      return;
    }
    const matches = byType(DEV_PLAY_TYPE[k]!);
    if (matches.length === 1) act(matches[0].flat); // road building: parameterless
    else setChoice(matches); // monopoly / year of plenty: pick resources
  };

  const onBarButton = (type: string) => {
    const matches = byType(type);
    if (matches.length === 1) act(matches[0].flat);
    else setChoice(matches); // maritime trade: pick the exchange
  };

  const canAfford = (cost: ResourceKind[]): boolean => {
    if (!me) return false;
    const need: Partial<Record<ResourceKind, number>> = {};
    for (const r of cost) need[r] = (need[r] ?? 0) + 1;
    return (Object.entries(need) as [ResourceKind, number][]).every(([r, n]) => me.resources[r] >= n);
  };

  // Road Building's free roads arrive as ordinary build_road actions; when the
  // hand can't cover the cost the build must be free, so show no price.
  const costFor = (a: GameAction): ResourceKind[] | undefined => {
    const cost = BUILD_COSTS[a.type];
    if (cost && a.type === "build_road" && !canAfford(cost)) return undefined;
    return cost;
  };

  const barTitle = (type: string) => {
    const cost = BUILD_COSTS[type];
    return cost ? `${actionMeta(type).label} — costs ${cost.join(", ")}` : actionMeta(type).label;
  };

  // The status line doubles as the prompt for what to click.
  const turnLabel = soloHuman ? "Your turn" : `${playerName(status.acting_player)}'s turn`;
  const hint = status.terminal
    ? "Game over"
    : !status.your_turn
      ? `${playerName(status.acting_player)} is thinking…`
      : knightArming
        ? `${turnLabel} — click a tile for the robber`
        : (
            {
              setup_settlement: `${turnLabel} — click a corner to place a settlement`,
              setup_road: `${turnLabel} — click an edge to place a road`,
              discard: `${turnLabel} — click resource cards to discard`,
              move_robber: `${turnLabel} — click a tile to move the robber`,
            } as Record<string, string>
          )[status.phase] ?? turnLabel;

  return (
    <div style={{ display: "flex", width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Board area: the chrome inside is anchored to it, not the viewport */}
      <div style={{ position: "relative", flex: 1, overflow: "hidden" }}>
        <BoardView board={board} interaction={interaction} />
        <TopBar mode="Play">
          <button style={smallButton} onClick={() => setConfiguring(true)}>
            New game
          </button>
        </TopBar>

        {popup && (
          <BoardPopover x={popup.x} y={popup.y} onClose={() => setPopup(null)}>
            {popup.actions.map((a) => {
              const cost = costFor(a);
              return (
                <button
                  key={a.flat}
                  disabled={busy}
                  style={{ ...buttonStyle, display: "flex", alignItems: "center", gap: 8, whiteSpace: "nowrap" }}
                  onClick={() => {
                    setPopup(null);
                    act(a.flat);
                  }}
                >
                  <span style={{ fontSize: 16 }}>{actionMeta(a.type).icon}</span>
                  <span>{popupLabel(a)}</span>
                  {cost && <CostRow cost={cost} />}
                </button>
              );
            })}
          </BoardPopover>
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
              color: ACCENT,
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
            {me && (
              <Hand
                player={me}
                you={soloHuman}
                discardable={discardable}
                onDiscard={onDiscard}
                playableDev={playableDev}
                armedDev={knightArming ? "knight" : null}
                onDev={onDev}
              />
            )}

            {/* Status + the turn-flow buttons share one row */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                fontSize: 13,
                flexWrap: "wrap",
                ...(me ? { borderTop: `1px solid ${DIVIDER}`, paddingTop: 8 } : {}),
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
              <span style={{ opacity: 0.6 }}>{hint}</span>
              {!status.terminal &&
                status.your_turn &&
                BAR_TYPES.filter((t) => availableTypes.has(t)).map((type) => (
                  <button
                    key={type}
                    title={barTitle(type)}
                    style={{ ...buttonStyle, fontSize: 13, padding: "6px 12px", whiteSpace: "nowrap" }}
                    disabled={busy}
                    onClick={() => onBarButton(type)}
                  >
                    {actionMeta(type).icon} {actionMeta(type).label}
                  </button>
                ))}
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
        players={board.players}
        acting={status.terminal ? undefined : status.acting_player}
        you={handSeat >= 0 ? handSeat : undefined}
        belief={snapshot.belief}
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
