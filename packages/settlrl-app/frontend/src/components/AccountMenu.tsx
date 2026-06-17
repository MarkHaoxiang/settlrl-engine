import { Link } from "react-router-dom";
import { logout, type AuthUser } from "../lib/auth";
import { LINK, smallButtonStyle } from "../lib/ui";

// A compact account control for the menu. Accounts are optional — signed out,
// everything still works — so signed out this is just a link to the sign-in
// page; signed in it shows the account and a log-out button.
export default function AccountMenu({
  user,
  onUser,
}: {
  user: AuthUser | null;
  onUser: (user: AuthUser | null) => void;
}) {
  if (user) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
        <span style={{ opacity: 0.8 }}>
          {user.email}
          {user.is_superuser ? " · admin" : ""}
        </span>
        <button style={smallButtonStyle} onClick={() => void logout().then(() => onUser(null))}>
          Log out
        </button>
      </div>
    );
  }

  return (
    <Link to="/login" style={{ ...smallButtonStyle, color: LINK, textDecoration: "none" }}>
      Sign in
    </Link>
  );
}
