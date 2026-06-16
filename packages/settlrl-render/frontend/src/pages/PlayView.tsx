import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import BoardView, {
  type BoardInteraction,
  type BoardTargetPoint,
  type TradeTargets,
} from "../components/BoardView";
import BoardPopover from "../components/BoardPopover";
import ChatPanel from "../components/ChatPanel";
import ChoicePopover from "../components/ChoicePopover";
import GameOverScreen from "../components/GameOverScreen";
import Hand, { DEV_PLAY_TYPE } from "../components/Hand";
import MaritimePopover from "../components/MaritimePopover";
import NewGameDialog from "../components/NewGameDialog";
import TradePopover from "../components/TradePopover";
import TopBar from "../components/TopBar";
import { useGame } from "../lib/useGame";
import { BUILD_COSTS, actionMeta } from "../lib/actionMeta";
import { createGame, joinGame, type GameAction, type GameSnapshot, type NewGameConfig } from "../lib/game";
import { deriveTransfers, tradeTransfer, type FlyToken } from "../lib/transfers";
import {
  parseTokens,
  rememberGame,
  resumeLink,
  saveTokens,
  tokensFor,
  type SeatTokens,
} from "../lib/seats";
import { PLAYER_COLORS, playerName, type DevCardKind, type ResourceKind } from "../lib/boardData";
import { cubeEq, edgeEq, hexEq, type Hex } from "../lib/hex";
import { ACCENT, DIVIDER, buttonStyle, overlayMsgStyle, panelStyle, smallButtonStyle } from "../lib/ui";

// Board-targeted action types, by the target geometry they carry. These are
// always marked directly on the board — no arming step.
const VERTEX_KIND: Record<string, "settlement" | "city"> = {
  setup_settlement: "settlement",
  build_settlement: "settlement",
  build_city: "city",
};
const EDGE_TYPES = new Set(["setup_road", "build_road"]);

// Turn-flow actions that stay on the bottom bar (display order). The rest
// happen on the table: trading on the bank piles and opponents' hand piles,
// buying a dev card on the bank deck, ending the turn here.
const BAR_TYPES = ["accept_trade", "reject_trade", "end_turn"];

const PHASE_LABEL: Record<string, string> = {
  setup_settlement: "Setup",
  setup_road: "Setup",
  roll: "Roll",
  discard: "Discard",
  move_robber: "Robber",
  main: "Main",
  trade_response: "Trade",
  game_over: "Game over",
};

export default function PlayView() {
  const { id: gameId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  // The seats this browser owns in this game (multiplayer identity).
  const [tokens, setTokens] = useState<SeatTokens>(() => (gameId ? tokensFor(gameId) : {}));
  const [joinFailed, setJoinFailed] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  // One join request at a time: the token effect above re-fires this one
  // before the first claim resolves, and a double-join would grab two seats.
  const joining = useRef(false);
  // A resume link (?tokens=seat:token,...) restores held seats, then is
  // stripped from the URL so it isn't re-applied or shared on. Declared before
  // the auto-join effect so storage holds the seat before that effect's guard.
  useEffect(() => {
    if (!gameId) return;
    const raw = searchParams.get("tokens");
    if (!raw) return;
    const incoming = parseTokens(raw);
    if (Object.keys(incoming).length > 0) {
      saveTokens(gameId, incoming);
      setTokens(tokensFor(gameId));
    }
    searchParams.delete("tokens");
    setSearchParams(searchParams, { replace: true });
  }, [gameId, searchParams, setSearchParams]);
  useEffect(() => {
    setTokens(gameId ? tokensFor(gameId) : {});
    setJoinFailed(false);
    setEndDismissed(false);
    joining.current = false;
    if (gameId) rememberGame(gameId);
  }, [gameId]);
  // Deep-linked with no claim: take the first free human seat, else spectate.
  useEffect(() => {
    if (!gameId || joinFailed || joining.current || Object.keys(tokens).length > 0) return;
    // The tokens state is stale in the render right after navigating from
    // start(); read storage directly so a creator's own fresh claims never
    // trigger a join (which would grab an invitee's seat).
    if (Object.keys(tokensFor(gameId)).length > 0) return;
    joining.current = true;
    joinGame(gameId).then(
      (j) => {
        saveTokens(gameId, { [j.seat]: j.token });
        joining.current = false;
        setTokens(tokensFor(gameId));
      },
      () => {
        joining.current = false;
        setJoinFailed(true);
      }
    );
  }, [gameId, tokens, joinFailed]);

  const { snapshot, error, busy, act, chat } = useGame(gameId ?? null, tokens);
  const mySeats = useMemo(
    () => Object.keys(tokens).map(Number).sort((a, b) => a - b),
    [tokens]
  );
  // The chooser anchored to a clicked board target.
  const [popup, setPopup] = useState<{ actions: GameAction[]; x: number; y: number } | null>(null);
  // The bottom-panel resource chooser (monopoly / year of plenty / trade).
  const [choice, setChoice] = useState<GameAction[] | null>(null);
  // Knight targeting: set while the knight chip awaits its robber tile.
  const [knightArming, setKnightArming] = useState(false);
  // The trade offer being composed, anchored at the partner's hand pile.
  const [tradeWith, setTradeWith] = useState<{ partner: number; x: number; y: number } | null>(null);
  // The maritime-trade picker, anchored at the clicked bank pile (the resource
  // to receive); choose which resource to give for one of it.
  const [maritimeFor, setMaritimeFor] = useState<{ receive: ResourceKind; x: number; y: number } | null>(null);
  // Whether the new-game configuration dialog is open (shown on entry
  // without a game, and reopened by the New game button).
  const [configuring, setConfiguring] = useState(!gameId);
  // The end-game overlay shows once a game finishes; "View board" dismisses it.
  const [endDismissed, setEndDismissed] = useState(false);
  // Set while waiting in line when the server is at its concurrency cap.
  const [queue, setQueue] = useState<{ position: number; total: number } | null>(null);
  const queueTimer = useRef<number | null>(null);

  // Creating a game claims human seats per the dialog's seating choice
  // (hotseat: all of them; online: just the first) — the rest are claimed
  // through the invite link (auto-join above). When the server is full the
  // request comes back as a queue position; we re-poll with the ticket until a
  // slot frees and the game is created.
  const QUEUE_POLL_MS = 3000;
  const pollCreate = async (config: NewGameConfig, ticket?: string) => {
    try {
      const res = await createGame(config, ticket);
      if ("queued" in res) {
        setQueue({ position: res.position, total: res.total });
        queueTimer.current = window.setTimeout(
          () => void pollCreate(config, res.ticket),
          QUEUE_POLL_MS
        );
      } else {
        setQueue(null);
        saveTokens(res.id, res.tokens);
        rememberGame(res.id);
        navigate(`/play/${res.id}`);
      }
    } catch (e) {
      setQueue(null);
      setCreateError(`Could not create the game: ${String(e)}`);
    }
  };
  const start = (config: NewGameConfig) => {
    setConfiguring(false);
    setCreateError(null);
    void pollCreate(config);
  };
  const cancelQueue = () => {
    if (queueTimer.current !== null) window.clearTimeout(queueTimer.current);
    queueTimer.current = null;
    setQueue(null);
    setConfiguring(true);
  };
  // Stop polling if the view unmounts mid-queue.
  useEffect(
    () => () => {
      if (queueTimer.current !== null) window.clearTimeout(queueTimer.current);
    },
    []
  );

  const actions = snapshot?.actions ?? [];

  // Reset transient UI when a new snapshot arrives.
  useEffect(() => {
    setPopup(null);
    setChoice(null);
    setKnightArming(false);
    setTradeWith(null);
    setMaritimeFor(null);
  }, [snapshot]);

  // Card-transfer animations: read the headline motions from each single-step
  // advance. A version gap (reconnect, or coalesced renders) can't be pinned to
  // one move, so we skip it rather than fly a misleading batch.
  const [transfers, setTransfers] = useState<FlyToken[]>([]);
  const prevSnap = useRef<GameSnapshot | null>(null);
  useEffect(() => {
    const prev = prevSnap.current;
    prevSnap.current = snapshot ?? null;
    if (prev && snapshot && snapshot.version === prev.version + 1) {
      const key = String(snapshot.version);
      // A trade resolves the offer pending on the prior snapshot; its accept
      // move is the latest log line on this one.
      const accepted =
        snapshot.log[snapshot.log.length - 1]?.action_type === "accept_trade";
      const t = [
        ...deriveTransfers(prev.board, snapshot.board, key),
        ...tradeTransfer(prev.status.trade, accepted, key),
      ];
      if (t.length) setTransfers(t);
    }
  }, [snapshot]);

  // Esc closes the choosers / cancels knight targeting.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPopup(null);
        setChoice(null);
        setKnightArming(false);
        setTradeWith(null);
        setMaritimeFor(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const byType = (type: string) => actions.filter((a) => a.type === type);
  const availableTypes = useMemo(() => new Set(actions.map((a) => a.type)), [actions]);

  // Board targets for every placeable action, all live at once. Knight tiles
  // only appear while the knight chip is armed — always-on they'd flood the
  // board any turn the card is in hand.
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

  if (!gameId)
    return (
      <div style={{ width: "100vw", height: "100vh" }}>
        {createError && <div style={overlayMsgStyle}>{createError}</div>}
        {queue && (
          <div style={overlayMsgStyle}>
            You're #{queue.position} of {queue.total} in line…
            <div style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
              The server is busy; your game starts automatically.
            </div>
            <button style={{ ...smallButtonStyle, marginTop: 12 }} onClick={cancelQueue}>
              Cancel
            </button>
          </div>
        )}
        {configuring && (
          <NewGameDialog onStart={(c) => void start(c)} onClose={() => navigate("/")} />
        )}
      </div>
    );
  if (error) return <div style={overlayMsgStyle}>{error}</div>;
  if (!snapshot) return <div style={overlayMsgStyle}>Loading game…</div>;

  const { status, board } = snapshot;
  // The hand panel follows whichever owned seat is acting (falling back to
  // this client's first seat). Owning no seats means spectating: no hand.
  const handSeat = mySeats.includes(status.acting_player)
    ? status.acting_player
    : (mySeats[0] ?? -1);
  const soloSeat = mySeats.length === 1;
  const me = handSeat >= 0 ? board.players[handSeat] : null;
  const winnerLabel =
    status.winner == null
      ? ""
      : mySeats.includes(status.winner)
        ? soloSeat
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

  // Arm / disarm knight targeting (the robber pawn and the knight hand chip
  // are the two ways in; the legal tiles then light up).
  const toggleKnight = () => {
    setKnightArming((v) => !v);
    setPopup(null);
  };

  const onDev = (k: DevCardKind) => {
    if (k === "knight") {
      toggleKnight();
      return;
    }
    const matches = byType(DEV_PLAY_TYPE[k]!);
    if (matches.length === 1) act(matches[0].flat); // road building: parameterless
    else setChoice(matches); // monopoly / year of plenty: pick resources
  };

  const onBarButton = (type: string) => {
    const matches = byType(type);
    if (matches.length === 1) act(matches[0].flat);
    else setChoice(matches); // trades: pick the exchange
  };

  const canAfford = (cost: ResourceKind[]): boolean => {
    if (!me?.resources) return false;
    const need: Partial<Record<ResourceKind, number>> = {};
    for (const r of cost) need[r] = (need[r] ?? 0) + 1;
    const hand = me.resources;
    return (Object.entries(need) as [ResourceKind, number][]).every(([r, n]) => hand[r] >= n);
  };

  // Road Building's free roads arrive as ordinary build_road actions; when the
  // hand can't cover the cost the build must be free, so show no price.
  const costFor = (a: GameAction): ResourceKind[] | undefined => {
    const cost = BUILD_COSTS[a.type];
    if (cost && a.type === "build_road" && !canAfford(cost)) return undefined;
    return cost;
  };

  // Rolling happens on the table: the dice glow and take the click.
  const rollAction = byType("roll_dice")[0];

  // So does trading: bank piles open the maritime exchanges for that
  // resource, opponents' hand piles open the 1:1 offer composer.
  const maritime = byType("maritime_trade");
  const proposals = byType("propose_trade");
  const tradeTargets: TradeTargets | undefined =
    maritime.length > 0 || proposals.length > 0
      ? {
          bank: new Set(maritime.map((a) => a.receive as ResourceKind)),
          partners: new Set(proposals.map((a) => a.partner as number)),
          onBank: (r, at) => setMaritimeFor({ receive: r, x: at.x, y: at.y }),
          onPartner: (p, at) => setTradeWith({ partner: p, x: at.x, y: at.y }),
        }
      : undefined;

  // Clicking the robber pawn is the other way to play a knight: it arms the
  // same targeting the knight hand chip does.
  const robberControl =
    !status.terminal && status.your_turn && !busy && availableTypes.has("play_knight")
      ? { armable: true, armed: knightArming, onToggle: toggleKnight }
      : undefined;

  // Buying a development card happens on the bank deck: clicking it confirms
  // the purchase (with its cost) in a board popover.
  const buyDevActions = byType("buy_development_card");
  const onBuyDev =
    !status.terminal && status.your_turn && !busy && buyDevActions.length > 0
      ? (at: BoardTargetPoint) => setPopup({ actions: buyDevActions, x: at.x, y: at.y })
      : undefined;

  const barTitle = (type: string) => {
    const cost = BUILD_COSTS[type];
    return cost ? `${actionMeta(type).label} — costs ${cost.join(", ")}` : actionMeta(type).label;
  };

  // The status line doubles as the prompt for what to click.
  const turnLabel = soloSeat && status.your_turn ? "Your turn" : `${playerName(status.acting_player)}'s turn`;
  const hint = status.terminal
    ? "Game over"
    : !status.your_turn
      ? `${mySeats.length === 0 ? "Spectating — " : ""}${playerName(status.acting_player)} is thinking…`
      : knightArming
        ? `${turnLabel} — click a tile for the robber`
        : (
            {
              setup_settlement: `${turnLabel} — click a corner to place a settlement`,
              setup_road: `${turnLabel} — click an edge to place a road`,
              discard: `${turnLabel} — click resource cards to discard`,
              move_robber: `${turnLabel} — click a tile to move the robber`,
              roll: `${turnLabel} — click the dice to roll`,
              trade_response: status.trade
                ? `${turnLabel} — ${playerName(status.trade.proposer)} offers ${status.trade.give} for your ${status.trade.receive}`
                : `${turnLabel} — accept or reject the trade`,
            } as Record<string, string>
          )[status.phase] ?? turnLabel;

  return (
    <div style={{ display: "flex", width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Board area: the chrome inside is anchored to it, not the viewport */}
      <div style={{ position: "relative", flex: 1, overflow: "hidden" }}>
        <BoardView
          board={board}
          interaction={interaction}
          trade={tradeTargets}
          robber={robberControl}
          onBuyDev={onBuyDev}
          transfers={transfers}
          dice={{
            sum: status.dice_roll,
            seed: snapshot.log.length,
            onRoll: rollAction && !busy ? () => act(rollAction.flat) : undefined,
            seat: status.terminal ? undefined : status.current_player,
          }}
        />
        <TopBar mode="Play">
          <button
            style={smallButtonStyle}
            title="Copy the invite link (others join free human seats)"
            onClick={() => void navigator.clipboard.writeText(window.location.href)}
          >
            🔗
          </button>
          {gameId && mySeats.length > 0 && (
            <button
              style={smallButtonStyle}
              title="Copy a link that restores your seat on another device or after clearing storage"
              onClick={() => void navigator.clipboard.writeText(resumeLink(gameId, tokens))}
            >
              🔑
            </button>
          )}
          <button style={smallButtonStyle} onClick={() => setConfiguring(true)}>
            New game
          </button>
        </TopBar>

        {tradeWith && me && (
          <TradePopover
            partner={tradeWith.partner}
            actions={proposals.filter((a) => a.partner === tradeWith.partner)}
            me={me}
            bounds={snapshot.belief?.players.find((b) => b.player === tradeWith.partner)}
            x={tradeWith.x}
            y={tradeWith.y}
            disabled={busy}
            onPick={(flat) => {
              setTradeWith(null);
              act(flat);
            }}
            onClose={() => setTradeWith(null)}
          />
        )}

        {maritimeFor && (
          <MaritimePopover
            receive={maritimeFor.receive}
            actions={maritime.filter((a) => a.receive === maritimeFor.receive)}
            board={board}
            player={status.acting_player}
            x={maritimeFor.x}
            y={maritimeFor.y}
            disabled={busy}
            onPick={(flat) => {
              setMaritimeFor(null);
              act(flat);
            }}
            onClose={() => setMaritimeFor(null)}
          />
        )}

        {popup && (
          <BoardPopover
            x={popup.x}
            y={popup.y}
            actions={popup.actions}
            costFor={costFor}
            disabled={busy}
            onPick={(flat) => {
              setPopup(null);
              act(flat);
            }}
            onClose={() => setPopup(null)}
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
              color: ACCENT,
              fontWeight: 700,
              zIndex: 10,
              cursor: endDismissed ? "pointer" : "default",
            }}
            onClick={endDismissed ? () => setEndDismissed(false) : undefined}
            title={endDismissed ? "Show final standings" : undefined}
          >
            {winnerLabel}
          </div>
        )}
        {status.terminal && !endDismissed && (
          <GameOverScreen
            board={board}
            winner={status.winner ?? null}
            mySeats={mySeats}
            onNewGame={() => setConfiguring(true)}
            onDismiss={() => setEndDismissed(true)}
          />
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
                you={soloSeat}
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
        <NewGameDialog onStart={(c) => void start(c)} onClose={() => setConfiguring(false)} />
      )}
    </div>
  );
}
