import { Link } from "react-router-dom";
import ThemeToggle from "../components/ThemeToggle";
import { panelStyle } from "../lib/ui";

const cardStyle: React.CSSProperties = {
  ...panelStyle,
  display: "flex",
  flexDirection: "column",
  gap: 8,
  width: 240,
  padding: "26px 24px",
  borderRadius: 16,
  border: "2px solid var(--panel-border)",
  textDecoration: "none",
  boxShadow: "0 6px 24px rgba(0,0,0,0.4)",
};

function MenuCard({ to, title, subtitle }: { to: string; title: string; subtitle: string }) {
  return (
    <Link to={to} style={cardStyle}>
      <span style={{ fontSize: 22, fontWeight: 700 }}>{title}</span>
      <span style={{ fontSize: 13, opacity: 0.75, lineHeight: 1.4 }}>{subtitle}</span>
    </Link>
  );
}

export default function Menu() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 36,
        padding: 24,
        color: "var(--text)",
        fontFamily: "Georgia, serif",
      }}
    >
      <h1 style={{ fontSize: 48, margin: 0, letterSpacing: 1 }}>
        Settlrl
      </h1>
      <div style={{ position: "fixed", top: 16, right: 16 }}>
        <ThemeToggle />
      </div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center" }}>
        <MenuCard to="/play" title="Play" subtitle="Start a new game and take turns on the board." />
        <MenuCard to="/replay" title="Replay" subtitle="Step through a recorded game from start to finish." />
      </div>
    </div>
  );
}
