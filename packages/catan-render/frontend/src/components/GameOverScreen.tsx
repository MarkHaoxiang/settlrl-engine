// The end-of-game overlay: winner headline plus final standings (public VP,
// awards, and piece/card counts). Dismissable to inspect the final board.

import type { Board, Player } from "../lib/boardData";
import { PLAYER_COLORS, playerName } from "../lib/boardData";
import { ACCENT, DIVIDER, buttonStyle, panelStyle, selectedStyle } from "../lib/ui";

const WIN_VP = 10;

// Publicly-visible victory points: building points + the two awards. Hidden
// victory-point dev cards are never revealed, so a winner can read above this;
// their row is floored at the win threshold.
const publicVp = (p: Player) => p.victoryPoints + (p.longestRoad ? 2 : 0) + (p.largestArmy ? 2 : 0);

export default function GameOverScreen({
  board,
  winner,
  mySeats,
  onNewGame,
  onDismiss,
}: {
  board: Board;
  winner: number | null;
  mySeats: number[];
  onNewGame: () => void;
  onDismiss: () => void;
}) {
  const vp = (p: Player) => (p.player === winner ? Math.max(publicVp(p), WIN_VP) : publicVp(p));
  const standings = [...board.players].sort((a, b) =>
    a.player === winner ? -1 : b.player === winner ? 1 : vp(b) - vp(a)
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
    <div
      style={{
        position: "absolute",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 30,
      }}
    >
      <div style={{ ...panelStyle, padding: "22px 26px", minWidth: 460, display: "flex", flexDirection: "column", gap: 16 }}>
        <span style={{ fontSize: 22, fontWeight: 800, color: ACCENT, textAlign: "center" }}>{headline}</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {standings.map((p) => (
            <div
              key={p.player}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${DIVIDER}`,
                ...(p.player === winner ? selectedStyle : {}),
              }}
            >
              <span style={{ width: 18, textAlign: "center" }}>{p.player === winner ? "👑" : ""}</span>
              <span style={{ width: 12, height: 12, borderRadius: "50%", background: PLAYER_COLORS[p.player] }} />
              <span style={{ flex: 1, fontWeight: 600 }}>{playerName(p.player)}</span>
              <span style={{ fontSize: 20, fontWeight: 800, width: 32, textAlign: "right" }}>{vp(p)}</span>
              <span style={{ width: 64, fontSize: 14, textAlign: "right" }} title="awards">
                {p.longestRoad ? "🛣️" : ""}
                {p.largestArmy ? `⚔️${p.knightsPlayed}` : ""}
              </span>
              <span style={{ width: 130, fontSize: 12, opacity: 0.75, textAlign: "right" }} title="settlements · cities · roads · cards · dev">
                🏠{pieces(p.player, "settlement")} 🏙{pieces(p.player, "city")} 🛤{roads(p.player)} · 🃏{p.resourceCards} 🎴{p.devCards}
              </span>
            </div>
          ))}
        </div>
        <span style={{ fontSize: 11, opacity: 0.55, textAlign: "center" }}>
          Points shown are public (building + awards); hidden victory-point cards aren't revealed.
        </span>
        <div style={{ display: "flex", justifyContent: "center", gap: 10 }}>
          <button style={buttonStyle} onClick={onDismiss}>
            View board
          </button>
          <button style={{ ...buttonStyle, ...selectedStyle }} onClick={onNewGame}>
            New game
          </button>
        </div>
      </div>
    </div>
  );
}
