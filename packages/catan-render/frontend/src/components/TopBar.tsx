import { Link } from "react-router-dom";
import { panelStyle } from "../lib/ui";

// The top-centre bar shared by the game views: a back-to-menu link and the
// current mode label. Pure presentation.
export default function TopBar({ mode }: { mode: string }) {
  return (
    <div
      style={{
        ...panelStyle,
        position: "absolute",
        top: 16,
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "6px 14px",
        zIndex: 10,
      }}
    >
      <Link to="/" style={{ color: "#9ec5e8", textDecoration: "none", fontSize: 14 }}>
        ← Menu
      </Link>
      <span style={{ fontWeight: 700, fontSize: 14, textTransform: "uppercase", letterSpacing: 1 }}>
        {mode}
      </span>
    </div>
  );
}
