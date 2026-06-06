// Modal dialog configuring a new game: players, number placement, seed.

import { useEffect, useState } from "react";
import type { NewGameConfig, NumberPlacement, PlayerCount } from "../lib/game";
import { buttonStyle, panelStyle, selectedStyle } from "../lib/ui";

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  opacity: 0.6,
  textTransform: "uppercase",
  letterSpacing: 1,
  width: 80,
};

// A labelled row of toggle buttons, one per option.
function Toggle<T extends string | number>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={labelStyle}>{label}</span>
      {options.map((o) => (
        <button
          key={o}
          style={{ ...buttonStyle, padding: "5px 12px", fontSize: 12, ...(value === o ? selectedStyle : {}) }}
          onClick={() => onChange(o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

export default function NewGameDialog({
  onStart,
  onClose,
}: {
  onStart: (config: NewGameConfig) => void;
  onClose: () => void;
}) {
  const [nPlayers, setNPlayers] = useState<PlayerCount>(4);
  const [numberPlacement, setNumberPlacement] = useState<NumberPlacement>("random");
  const [seed, setSeed] = useState("");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const start = () =>
    onStart({
      seed: seed === "" ? Math.floor(Math.random() * 65536) : Number(seed),
      nPlayers,
      numberPlacement,
    });

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{ ...panelStyle, display: "flex", flexDirection: "column", gap: 14, padding: "20px 24px", minWidth: 300 }}
        onClick={(e) => e.stopPropagation()}
      >
        <span style={{ fontSize: 18, fontWeight: 700 }}>New game</span>
        <Toggle label="Players" options={[2, 4] as const} value={nPlayers} onChange={setNPlayers} />
        <Toggle
          label="Numbers"
          options={["random", "spiral"] as const}
          value={numberPlacement}
          onChange={setNumberPlacement}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={labelStyle}>Seed</span>
          <input
            type="number"
            placeholder="random"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            style={{ ...buttonStyle, cursor: "text", width: 100, padding: "5px 10px", fontSize: 12 }}
          />
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button style={buttonStyle} onClick={onClose}>
            Cancel
          </button>
          <button style={{ ...buttonStyle, ...selectedStyle }} onClick={start}>
            Start
          </button>
        </div>
      </div>
    </div>
  );
}
