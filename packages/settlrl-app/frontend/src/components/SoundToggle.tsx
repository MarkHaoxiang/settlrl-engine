import { useState } from "react";
import { isMuted, setMuted } from "../lib/sound";
import { buttonStyle } from "../lib/ui";

// Mute / unmute the action sound effects. Off by default; the choice persists
// in localStorage (see lib/sound). Local state just keeps the icon current.
export default function SoundToggle() {
  const [muted, setMutedState] = useState(isMuted());
  return (
    <button
      title={muted ? "Turn sound on" : "Turn sound off"}
      style={{ ...buttonStyle, padding: "3px 8px", fontSize: 13, lineHeight: 1.2 }}
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
