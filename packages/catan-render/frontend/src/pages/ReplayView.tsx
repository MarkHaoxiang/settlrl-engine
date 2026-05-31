import { useParams } from "react-router-dom";
import GameShell from "../components/GameShell";

const barStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 14,
  padding: "10px 18px",
  borderRadius: 14,
  background: "rgba(12, 28, 46, 0.82)",
  border: "1px solid rgba(255,255,255,0.15)",
  color: "#F2EFE6",
  fontFamily: "Georgia, serif",
  backdropFilter: "blur(2px)",
  userSelect: "none",
};

const btnStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.2)",
  color: "#F2EFE6",
  borderRadius: 8,
  width: 36,
  height: 32,
  fontSize: 14,
  cursor: "pointer",
};

// Playback controls for stepping through a recorded game. These are presentation
// stubs for now — wiring them to a backend that serves recorded game states is
// future work (e.g. GET /api/replay/:id/step/:n).
function ReplayControls({ gameId }: { gameId?: string }) {
  return (
    <div style={barStyle}>
      <span style={{ fontSize: 12, opacity: 0.7 }}>{gameId ? `game ${gameId}` : "demo"}</span>
      <button style={btnStyle} title="Step back" disabled>
        ◀
      </button>
      <button style={btnStyle} title="Play / pause" disabled>
        ▮▮
      </button>
      <button style={btnStyle} title="Step forward" disabled>
        ▶
      </button>
      <input type="range" min={0} max={100} defaultValue={0} disabled style={{ width: 280 }} />
      <span style={{ fontSize: 11, opacity: 0.6 }}>turn 0 / 0</span>
    </div>
  );
}

export default function ReplayView() {
  const { gameId } = useParams();
  return <GameShell mode="Replay" controls={() => <ReplayControls gameId={gameId} />} />;
}
