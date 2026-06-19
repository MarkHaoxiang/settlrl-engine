// The end-of-game overlay: winner headline plus final standings (victory
// points, awards, and piece/card counts). Dismissable to inspect the board.

import type { Board, Player } from "../lib/boardData";
import { PLAYER_COLORS, playerName } from "../lib/boardData";
import { downloadRecord } from "../lib/game";
import Button from "./Button";
import Modal from "./Modal";
import s from "./GameOverScreen.module.css";

// Final victory points: building points + the two awards + victory-point dev
// cards. Every hand is revealed once the game is over, so this is the true
// total for every player.
const totalVp = (p: Player) =>
  p.victoryPoints +
  (p.longestRoad ? 2 : 0) +
  (p.largestArmy ? 2 : 0) +
  (p.devCardTypes?.victory_point ?? 0);

export default function GameOverScreen({
  board,
  winner,
  mySeats,
  gameId,
  onNewGame,
  onDismiss,
}: {
  board: Board;
  winner: number | null;
  mySeats: number[];
  gameId?: string;
  onNewGame: () => void;
  onDismiss: () => void;
}) {
  const standings = [...board.players].sort((a, b) =>
    a.player === winner ? -1 : b.player === winner ? 1 : totalVp(b) - totalVp(a)
  );
  const headline =
    winner == null
      ? "Game over"
      : mySeats.includes(winner)
        ? "You win! 🎉"
        : `${playerName(winner)} wins! 🎉`;

  const pieces = (player: number, kind: "settlement" | "city") =>
    board.buildings.filter((b) => b.player === player && b.kind === kind).length;
  const roads = (player: number) => board.roads.filter((r) => r.player === player).length;

  return (
    <Modal onClose={onDismiss} title={headline}>
      <div className={s.dialog}>
        <span className={s.headline}>{headline}</span>
        <div className={s.standings}>
          {standings.map((p) => (
            <div key={p.player} className={p.player === winner ? s.rowWin : s.row}>
              <span className={s.crown}>{p.player === winner ? "👑" : ""}</span>
              <span className={s.dot} style={{ background: PLAYER_COLORS[p.player] }} />
              <span className={s.name}>{playerName(p.player)}</span>
              <span className={s.vp}>{totalVp(p)}</span>
              <span className={s.awards} title="awards">
                {p.longestRoad ? "🛣️" : ""}
                {p.largestArmy ? `⚔️${p.knightsPlayed}` : ""}
              </span>
              <span
                className={s.counts}
                title="settlements · cities · roads · cards · dev"
              >
                🏠{pieces(p.player, "settlement")} 🏙{pieces(p.player, "city")} 🛤
                {roads(p.player)} · 🃏{p.resourceCards} 🎴{p.devCards}
              </span>
            </div>
          ))}
        </div>
        <div className={s.actions}>
          <Button onClick={onDismiss}>View board</Button>
          {gameId && (
            <Button onClick={() => void downloadRecord(gameId)}>Download replay</Button>
          )}
          <Button selected onClick={onNewGame}>
            New game
          </Button>
        </div>
      </div>
    </Modal>
  );
}
