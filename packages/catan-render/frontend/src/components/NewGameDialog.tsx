// Modal dialog configuring a new game: players, seat controllers (with
// per-bot parameter overrides), number placement, seed.

import { useEffect, useState } from "react";
import {
  fetchBots,
  HUMAN,
  type BotParamValue,
  type BotSpec,
  type NewGameConfig,
  type NumberPlacement,
  type PlayerCount,
  type SeatConfig,
} from "../lib/game";
import { playerName } from "../lib/boardData";
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
  trailing,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  trailing?: React.ReactNode;
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
      {trailing}
    </div>
  );
}

// The parameter rows for one configured bot seat. Values not overridden show
// (and reset to) the catalog defaults; only overrides are sent to the server.
function SeatParams({
  spec,
  params,
  onChange,
}: {
  spec: BotSpec;
  params: Record<string, BotParamValue>;
  onChange: (params: Record<string, BotParamValue>) => void;
}) {
  const set = (name: string, value: BotParamValue | undefined) => {
    const next = { ...params };
    if (value === undefined || value === spec.params[name].default) delete next[name];
    else next[name] = value;
    onChange(next);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, margin: "0 0 4px 86px" }}>
      {Object.entries(spec.params).map(([name, p]) => (
        <div key={name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ ...labelStyle, width: 170, textTransform: "none" }}>{name}</span>
          {p.type === "bool" ? (
            <input
              type="checkbox"
              checked={Boolean(params[name] ?? p.default)}
              onChange={(e) => set(name, e.target.checked)}
            />
          ) : (
            <input
              type="number"
              step={p.type === "int" ? 1 : "any"}
              value={String(params[name] ?? p.default)}
              onChange={(e) => {
                const v = e.target.value === "" ? undefined : Number(e.target.value);
                set(name, v === undefined || Number.isNaN(v) ? undefined : p.type === "int" ? Math.round(v) : v);
              }}
              style={{ ...buttonStyle, cursor: "text", width: 80, padding: "3px 8px", fontSize: 12 }}
            />
          )}
        </div>
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
  // One controller per possible seat; only the first nPlayers are used.
  const [seats, setSeats] = useState<SeatConfig[]>([
    { kind: HUMAN },
    { kind: "random" },
    { kind: "random" },
    { kind: "random" },
  ]);
  const [open, setOpen] = useState<boolean[]>([false, false, false, false]);
  const [bots, setBots] = useState<Record<string, BotSpec>>({});

  useEffect(() => {
    fetchBots().then(setBots).catch(() => setBots({}));
  }, []);

  // Bot kinds available at the chosen player count; a seat holding a kind
  // that the new count doesn't support falls back to "random".
  const botNames = Object.keys(bots)
    .filter((b) => bots[b].counts.includes(nPlayers))
    .sort();
  useEffect(() => {
    setSeats((prev) =>
      prev.map((s) =>
        s.kind === HUMAN || bots[s.kind]?.counts.includes(nPlayers) ? s : { kind: "random" }
      )
    );
  }, [nPlayers, bots]);

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
      seats: seats.slice(0, nPlayers),
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
        {seats.slice(0, nPlayers).map((seat, i) => {
          const spec = bots[seat.kind];
          const hasKnobs = spec && Object.keys(spec.params).length > 0;
          const overridden = Object.keys(seat.params ?? {}).length > 0;
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <Toggle
                label={playerName(i)}
                options={[HUMAN, ...botNames]}
                value={seat.kind}
                onChange={(v) =>
                  setSeats((prev) => prev.map((p, j) => (j === i ? { kind: v } : p)))
                }
                trailing={
                  hasKnobs ? (
                    <button
                      title="Configure bot parameters"
                      style={{
                        ...buttonStyle,
                        padding: "5px 9px",
                        fontSize: 12,
                        ...(open[i] || overridden ? selectedStyle : {}),
                      }}
                      onClick={() => setOpen((prev) => prev.map((o, j) => (j === i ? !o : o)))}
                    >
                      {"⚙"}
                      {overridden ? "*" : ""}
                    </button>
                  ) : undefined
                }
              />
              {hasKnobs && open[i] && (
                <SeatParams
                  spec={spec}
                  params={seat.params ?? {}}
                  onChange={(params) =>
                    setSeats((prev) => prev.map((p, j) => (j === i ? { ...p, params } : p)))
                  }
                />
              )}
            </div>
          );
        })}
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
