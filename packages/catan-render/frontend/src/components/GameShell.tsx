import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useBoard } from "../lib/useBoard";
import type { Board } from "../lib/boardData";
import BoardView from "./BoardView";

interface Props {
  // Short label shown in the top bar (e.g. "Replay", "Play").
  mode: string;
  // Mode-specific controls, rendered in a bar anchored to the bottom centre.
  // Receives the loaded board so controls can reflect game state.
  controls: (board: Board) => ReactNode;
}

const overlayMsg: React.CSSProperties = { color: "#fff", padding: 24, fontFamily: "Georgia, serif" };

// Full-screen frame shared by the replay and play views: it loads the board,
// renders it via BoardView, and overlays a back-to-menu link plus a slot for
// mode-specific controls at the bottom.
export default function GameShell({ mode, controls }: Props) {
  const { board, error } = useBoard();

  if (error) return <div style={overlayMsg}>{error}</div>;
  if (!board) return <div style={overlayMsg}>Loading board…</div>;

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden" }}>
      <BoardView board={board} />

      {/* Top centre: mode label + exit to menu */}
      <div
        style={{
          position: "absolute",
          top: 16,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "6px 14px",
          borderRadius: 12,
          background: "rgba(12, 28, 46, 0.82)",
          border: "1px solid rgba(255,255,255,0.15)",
          color: "#F2EFE6",
          fontFamily: "Georgia, serif",
          backdropFilter: "blur(2px)",
          userSelect: "none",
        }}
      >
        <Link to="/" style={{ color: "#9ec5e8", textDecoration: "none", fontSize: 14 }}>
          ← Menu
        </Link>
        <span style={{ fontWeight: 700, fontSize: 14, textTransform: "uppercase", letterSpacing: 1 }}>
          {mode}
        </span>
      </div>

      {/* Bottom centre: mode-specific controls */}
      <div
        style={{
          position: "absolute",
          bottom: 20,
          left: "50%",
          transform: "translateX(-50%)",
        }}
      >
        {controls(board)}
      </div>
    </div>
  );
}
