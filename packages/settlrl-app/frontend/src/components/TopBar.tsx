import { Link } from "react-router-dom";
import SoundToggle from "./SoundToggle";
import ThemeToggle from "./ThemeToggle";
import s from "./TopBar.module.css";

// The top bar shared by the game views: a help link in the top-left corner,
// and a back-to-menu link with the current mode label top-centre, followed by
// the theme toggle and any view-specific controls (`children` — settings-like
// actions such as New game).
export default function TopBar({ mode, children }: { mode: string; children?: React.ReactNode }) {
  return (
    <>
      <Link to="/help" title="Help" className={s.help}>
        ?
      </Link>
      <div className={s.bar}>
        <Link to="/" className={s.menuLink}>
          ← Menu
        </Link>
        <span className={s.mode}>{mode}</span>
        {children}
        <SoundToggle />
        <ThemeToggle />
      </div>
    </>
  );
}
