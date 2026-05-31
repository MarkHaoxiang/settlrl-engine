import { Link } from "react-router-dom";

const cardStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  width: 240,
  padding: "26px 24px",
  borderRadius: 16,
  background: "rgba(12, 28, 46, 0.82)",
  border: "2px solid rgba(255,255,255,0.15)",
  color: "#F2EFE6",
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
        color: "#F2EFE6",
        fontFamily: "Georgia, serif",
      }}
    >
      <h1 style={{ fontSize: 48, margin: 0, letterSpacing: 1, textShadow: "0 2px 12px rgba(0,0,0,0.5)" }}>
        Catan
      </h1>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center" }}>
        <MenuCard to="/play" title="Play" subtitle="Start a new game and take turns on the board." />
        <MenuCard to="/replay" title="Replay" subtitle="Step through a recorded game from start to finish." />
      </div>
    </div>
  );
}
