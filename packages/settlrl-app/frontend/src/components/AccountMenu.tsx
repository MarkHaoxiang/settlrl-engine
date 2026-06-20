import { Link } from "react-router-dom";
import { logout, type AuthUser } from "../lib/auth";
import ui from "../styles/ui.module.css";
import Button from "./Button";
import { HumanIcon } from "./icons";
import s from "./AccountMenu.module.css";

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
      <div className={s.row}>
        <Link
          to="/profile"
          className={s.accountLink}
          title={`${user.email}${user.is_superuser ? " · admin" : ""} — your profile`}
          aria-label="Your profile"
        >
          <HumanIcon size={18} />
          {user.is_superuser ? <span className={s.adminTag}>admin</span> : null}
        </Link>
        <Button variant="small" onClick={() => void logout().then(() => onUser(null))}>
          Log out
        </Button>
      </div>
    );
  }

  return (
    <Link to="/login" className={ui.buttonLinkSmall}>
      Sign in
    </Link>
  );
}
