import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { currentUser } from "../lib/auth";
import s from "./AdminLink.module.css";

// A persistent corner link to the admin status page, shown only to superusers.
// Re-checks on navigation so it appears/disappears as the account changes.
export default function AdminLink() {
  const [admin, setAdmin] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    let live = true;
    void currentUser().then((u) => live && setAdmin(!!u?.is_superuser));
    return () => {
      live = false;
    };
  }, [pathname]);

  if (!admin || pathname === "/admin") return null;
  return (
    <Link to="/admin" className={s.link} title="Server status (admin)">
      ⚙ Admin
    </Link>
  );
}
