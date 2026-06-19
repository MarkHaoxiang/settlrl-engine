import { useState } from "react";
import { isMuted, setMuted } from "../lib/sound";
import ui from "../styles/ui.module.css";

// Mute / unmute the action sound effects. Off by default; the choice persists
// in localStorage (see lib/sound). Local state just keeps the icon current.
export default function SoundToggle() {
  const [muted, setMutedState] = useState(isMuted());
  return (
    <button
      title={muted ? "Turn sound on" : "Turn sound off"}
      className={ui.iconButton}
      onClick={() => {
        const next = !muted;
        setMuted(next);
        setMutedState(next);
      }}
    >
      {muted ? "🔇" : "🔊"}
    </button>
  );
}
