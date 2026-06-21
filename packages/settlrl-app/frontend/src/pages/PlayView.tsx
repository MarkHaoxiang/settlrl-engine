import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
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
import TradePopover from "../components/TradePopover";
import TradeResponsePopover from "../components/TradeResponsePopover";
import TopBar from "../components/TopBar";
import { BotIcon } from "../components/icons";
import { useGame } from "../lib/useGame";
import { BUILD_COSTS, actionMeta } from "../lib/actionMeta";
import { type GameAction, type GameSnapshot } from "../lib/game";
import { deriveTransfers, tradeTransfer, type FlyToken } from "../lib/transfers";
import { clearCurrentPlace, setCurrentPlace, tokensFor, type SeatTokens } from "../lib/seats";
import { PLAYER_COLORS, playerName, type DevCardKind, type ResourceKind } from "../lib/boardData";
import { cubeEq, edgeEq, hexEq, type Hex } from "../lib/hex";
import Button from "../components/Button";
import ui from "../styles/ui.module.css";
import s from "./PlayView.module.css";

// Board-targeted action types, by the target geometry they carry. These are
// always marked directly on the board — no arming step.
const VERTEX_KIND: Record<string, "settlement" | "city"> = {
  setup_settlement: "settlement",
  build_settlement: "settlement",
  build_city: "city",
};
const EDGE_TYPES = new Set(["setup_road", "build_road"]);

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
  // The seats this browser owns in this game (multiplayer identity).
  const [tokens, setTokens] = useState<SeatTokens>(() => (gameId ? tokensFor(gameId) : {}));
  useEffect(() => {
    setTokens(gameId ? tokensFor(gameId) : {});
    setEndDismissed(false);
  }, [gameId]);
  const { snapshot, error, busy, act, chat } = useGame(gameId ?? null, tokens);

  // The seats this client controls: the server's your_seats (token- or
  // account-owned) once a snapshot is in, else the local tokens.
  const mySeats = useMemo(
    () =>
      snapshot
        ? [...snapshot.your_seats].sort((a, b) => a - b)
        : Object.keys(tokens).map(Number).sort((a, b) => a - b),
    [snapshot, tokens]
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
  // The end-game overlay shows once a game finishes; "View board" dismisses it.
  const [endDismissed, setEndDismissed] = useState(false);

  // A game still waiting for players belongs in its lobby room, not on the board:
  // bounce there (this keeps old /play invite links working). Play only begins
  // once every human seat is claimed.
  useEffect(() => {
    if (!gameId || !snapshot) return;
    const st = snapshot.status;
    const claimed = new Set(snapshot.seats_claimed);
    const stillWaiting =
      !st.terminal && st.seats.some((k, i) => k === "human" && !claimed.has(i));
    if (stillWaiting) navigate(`/lobby/${gameId}`, { replace: true });
  }, [gameId, snapshot, navigate]);

  // Track this as the browser's current game (the one-game-at-a-time guard for
  // guests): held while it's live and we hold a seat, forgotten once it ends.
  useEffect(() => {
    if (!gameId || !snapshot || mySeats.length === 0) return;
    if (snapshot.status.terminal) clearCurrentPlace(gameId);
    else setCurrentPlace(gameId, "game");
  }, [gameId, snapshot, mySeats]);

  const actions = snapshot?.actions ?? [];

  // Dismiss every transient chooser / armed state at once (a new snapshot, Esc,
  // or an outside click) — one place so a new overlay can't be missed.
  const clearOverlays = useCallback(() => {
    setPopup(null);
    setChoice(null);
    setKnightArming(false);
    setTradeWith(null);
    setMaritimeFor(null);
  }, []);

  // Reset transient UI when a new snapshot arrives.
  useEffect(() => {
    clearOverlays();
  }, [snapshot, clearOverlays]);

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
      if (e.key === "Escape") clearOverlays();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearOverlays]);

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

  if (error) return <div className={ui.overlayMsg}>{error}</div>;
  if (!snapshot) return <div className={ui.overlayMsg}>Loading game…</div>;

  const { status, board } = snapshot;
  // Who holds each seat: an account name (or "Guest") for a human, the bot kind
  // for a bot — for the seat list in the chat column.
  const seatIdentity = (i: number): string =>
    status.seats[i] !== "human" ? status.seats[i] : (snapshot.seat_names[i] ?? "Guest");
  const seatLabels = status.seats.map((_, i) => seatIdentity(i));
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

  // Rolling happens on the table: the dice glow gold and take the click; once
  // rolling is done the same dice pulse red as the end-turn control.
  const rollAction = byType("roll_dice")[0];
  const endTurnAction = byType("end_turn")[0];

  // An incoming trade offer awaiting this client's answer (it owns the seat the
  // proposal was made to) shows as a floating card with Accept / Reject.
  const acceptAction = byType("accept_trade")[0];
  const rejectAction = byType("reject_trade")[0];
  const incomingTrade =
    !status.terminal && status.your_turn && status.phase === "trade_response" && status.trade
      ? status.trade
      : null;

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
    <div className={s.layout}>
      {/* Board area: the chrome inside is anchored to it, not the viewport */}
      <div className={s.boardArea}>
        <BoardView
          board={board}
          interaction={interaction}
          // Face the viewer's seat when they hold exactly one (online play);
          // a hotseat (many seats) or spectator keeps the canonical view.
          faceSeat={soloSeat ? mySeats[0] : undefined}
          trade={tradeTargets}
          robber={robberControl}
          onBuyDev={onBuyDev}
          transfers={transfers}
          dice={{
            sum: status.dice_roll,
            seed: snapshot.log.length,
            onRoll: rollAction && !busy ? () => act(rollAction.flat) : undefined,
            onEndTurn:
              endTurnAction && !busy && status.your_turn ? () => act(endTurnAction.flat) : undefined,
            seat: status.terminal ? undefined : status.current_player,
          }}
        />
        <TopBar mode="Play">
          <Button
            variant="small"
            title="Copy the invite link (others join free human seats)"
            onClick={() => void navigator.clipboard.writeText(window.location.href)}
          >
            🔗
          </Button>
          <Button variant="small" onClick={() => navigate("/lobby")}>
            New game
          </Button>
        </TopBar>

        {incomingTrade && (acceptAction || rejectAction) && (
          <TradeResponsePopover
            offer={incomingTrade}
            disabled={busy}
            onAccept={() => acceptAction && act(acceptAction.flat)}
            onReject={() => rejectAction && act(rejectAction.flat)}
          />
        )}

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
            className={s.winnerBanner}
            style={{ cursor: endDismissed ? "pointer" : "default" }}
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
            gameId={gameId}
            onNewGame={() => navigate("/lobby")}
            onDismiss={() => setEndDismissed(true)}
          />
        )}

        {/* Full-width flex strip so the panel centres without halving the
            shrink-to-fit width (which would wrap the hand). */}
        <div className={s.bottomStrip}>
          <div className={s.bottomPanel}>
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
            <div className={me ? s.statusRowDivided : s.statusRow}>
              <span className={s.phase}>{PHASE_LABEL[status.phase] ?? status.phase}</span>
              {status.victory_points_to_win !== 10 && (
                <span
                  className={s.winTarget}
                  title={`First to ${status.victory_points_to_win} victory points wins`}
                >
                  🏆 {status.victory_points_to_win}
                </span>
              )}
              {snapshot.bot_move && (
                <span
                  className={`fade-in ${s.botMove}`}
                  title={`${playerName(snapshot.bot_move.player)} · ${snapshot.bot_move.action.label}`}
                >
                  <span
                    className={s.botDot}
                    style={{ background: PLAYER_COLORS[snapshot.bot_move.player] ?? "#888" }}
                  />
                  <BotIcon size={15} />
                  <span className={s.botIcon}>{actionMeta(snapshot.bot_move.action.type).icon}</span>
                </span>
              )}
              <span className={s.hint}>{hint}</span>
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
        identities={seatLabels}
      />
    </div>
  );
}
