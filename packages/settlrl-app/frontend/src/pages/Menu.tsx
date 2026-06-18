import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import BotProviders from "../components/BotProviders";
import MyGames from "../components/MyGames";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
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
  const [user, setUser] = useState<AuthUser | null>(null);
  useEffect(() => {
    void currentUser().then(setUser);
  }, []);
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
      <div
        style={{
          position: "fixed",
          top: 16,
          right: 16,
          display: "flex",
          alignItems: "flex-start",
          gap: 10,
        }}
      >
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center" }}>
        <MenuCard to="/play" title="Play" subtitle="Start a new game and take turns on the board." />
        <MenuCard to="/replay" title="Replay" subtitle="Step through a recorded game from start to finish." />
        <MenuCard to="/leaderboard" title="Leaderboard" subtitle="Elo rankings for players and bots, split by game size." />
      </div>
      <MyGames user={user} />
      <BotProviders user={user} />
    </div>
  );
}
