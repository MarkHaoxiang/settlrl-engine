import { Link } from "react-router-dom";
import { panelStyle } from "../lib/ui";

// The top bar shared by the game views: a help link on the left (tucked under
// the top-left player panel), and a back-to-menu link with the current mode
// label top-centre. Pure presentation.
export default function TopBar({ mode }: { mode: string }) {
  return (
    <>
      <Link
        to="/help"
        title="Help"
        style={{
          ...panelStyle,
          position: "absolute",
          top: 110,
          left: 16,
          width: 32,
          height: 32,
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#9ec5e8",
          textDecoration: "none",
          fontSize: 16,
          fontWeight: 700,
          zIndex: 10,
        }}
      >
        ?
      </Link>
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
    </>
  );
}
