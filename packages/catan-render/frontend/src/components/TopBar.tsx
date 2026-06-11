import { Link } from "react-router-dom";
import { LINK, panelStyle } from "../lib/ui";
import ThemeToggle from "./ThemeToggle";

// The top bar shared by the game views: a help link on the left (tucked under
// the top-left player panel), and a back-to-menu link with the current mode
// label top-centre, followed by the theme toggle and any view-specific
// controls (`children` — settings-like actions such as New game).
export default function TopBar({ mode, children }: { mode: string; children?: React.ReactNode }) {
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
          color: LINK,
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
        <Link to="/" style={{ color: LINK, textDecoration: "none", fontSize: 14 }}>
          ← Menu
        </Link>
        <span style={{ fontWeight: 700, fontSize: 14, textTransform: "uppercase", letterSpacing: 1 }}>
          {mode}
        </span>
        {children}
        <ThemeToggle />
      </div>
    </>
  );
}
