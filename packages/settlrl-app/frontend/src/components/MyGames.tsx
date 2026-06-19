import { Link } from "react-router-dom";
import { type AuthUser } from "../lib/auth";
import { useMyGames } from "../lib/queries";
import Panel from "./Panel";
import s from "./MyGames.module.css";

// The signed-in user's in-progress games, so they can resume on any device
// (seats follow the account). Hidden when logged out or when there are none.
export default function MyGames({ user }: { user: AuthUser | null }) {
  const games = useMyGames(user).data ?? [];

  if (games.length === 0) return null;
  return (
    <Panel className={s.box}>
      <div className={s.label}>Your games</div>
      <div className={s.list}>
        {games.map((g) => (
          <Link key={g.id} to={`/play/${g.id}`} className={s.gameLink}>
            Game {g.id.slice(0, 6)} — seat{g.seats.length > 1 ? "s" : ""}{" "}
            {g.seats.map((s) => s + 1).join(", ")}
          </Link>
        ))}
      </div>
    </Panel>
  );
}
