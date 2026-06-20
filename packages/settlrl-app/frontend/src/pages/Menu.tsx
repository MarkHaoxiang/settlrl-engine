import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import AccountMenu from "../components/AccountMenu";
import MyGames from "../components/MyGames";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, type AuthUser } from "../lib/auth";
import s from "./Menu.module.css";

function MenuCard({ to, title, subtitle }: { to: string; title: string; subtitle: string }) {
  return (
    <Link to={to} className={s.card}>
      <span className={s.cardTitle}>{title}</span>
      <span className={s.cardSubtitle}>{subtitle}</span>
    </Link>
  );
}

export default function Menu() {
  const [user, setUser] = useState<AuthUser | null>(null);
  useEffect(() => {
    void currentUser().then(setUser);
  }, []);
  return (
    <div className={s.page}>
      <h1 className={s.title}>Settlrl</h1>
      <div className={s.toolbar}>
        <AccountMenu user={user} onUser={setUser} />
        <ThemeToggle />
      </div>
      <div className={s.cards}>
        <MenuCard to="/lobby" title="Play" subtitle="Host a game, join an open one, or quick match." />
        <MenuCard to="/replay" title="Replay" subtitle="Step through a recorded game from start to finish." />
        <MenuCard to="/leaderboard" title="Leaderboard" subtitle="Elo rankings for players and bots, split by game size." />
      </div>
      <MyGames user={user} />
    </div>
  );
}
