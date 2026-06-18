import { Link } from "react-router-dom";
import { type AuthUser } from "../lib/auth";
import { useMyGames } from "../lib/queries";
import { LINK, panelStyle } from "../lib/ui";

// The signed-in user's in-progress games, so they can resume on any device
// (seats follow the account). Hidden when logged out or when there are none.
export default function MyGames({ user }: { user: AuthUser | null }) {
  const games = useMyGames(user).data ?? [];

  if (games.length === 0) return null;
  return (
    <div style={{ ...panelStyle, padding: "16px 20px", borderRadius: 12, minWidth: 248 }}>
      <div
        style={{
          fontSize: 12,
          opacity: 0.6,
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 10,
        }}
      >
        Your games
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {games.map((g) => (
          <Link key={g.id} to={`/play/${g.id}`} style={{ color: LINK, fontSize: 14 }}>
            Game {g.id.slice(0, 6)} — seat{g.seats.length > 1 ? "s" : ""}{" "}
            {g.seats.map((s) => s + 1).join(", ")}
          </Link>
        ))}
      </div>
    </div>
  );
}
