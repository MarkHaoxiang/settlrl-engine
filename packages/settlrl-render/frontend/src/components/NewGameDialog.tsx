// Modal dialog configuring a new game: players, per-seat Human/Bot choice,
// number placement, seed. Choosing Bot opens an in-dialog picker listing each
// bot kind with a description and its tunable parameters.

import { useEffect, useMemo, useState } from "react";
import {
  fetchBots,
  fetchPreview,
  HUMAN,
  type BotParamValue,
  type BotSpec,
  type NewGameConfig,
  type NumberPlacement,
  type PlayerCount,
  type SeatConfig,
} from "../lib/game";
import { type Board, playerName } from "../lib/boardData";
import { buttonStyle, panelStyle, selectedStyle } from "../lib/ui";
import BoardView from "./BoardView";
import { BotIcon, HumanIcon, MapIcon } from "./icons";

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  opacity: 0.6,
  textTransform: "uppercase",
  letterSpacing: 1,
  width: 80,
};

// Seat Human/Bot buttons: an icon beside the label.
const seatButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  padding: "5px 12px",
  fontSize: 12,
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
};

const botLabel = (kind: string) =>
  kind === "mcts" ? "MCTS" : kind.charAt(0).toUpperCase() + kind.slice(1);

// A labelled row of toggle buttons, one per option. `optionTitle` adds per-
// option hover help.
function Toggle<T extends string | number>({
  label,
  options,
  value,
  onChange,
  trailing,
  optionTitle,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  trailing?: React.ReactNode;
  optionTitle?: (o: T) => string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={labelStyle}>{label}</span>
      {options.map((o) => (
        <button
          key={o}
          title={optionTitle?.(o)}
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

// Hover help for the number-placement options.
const PLACEMENT_HELP: Record<NumberPlacement, string> = {
  random: "Number tokens shuffled uniformly across the land tiles.",
  spiral: "The rulebook's variable setup: tokens A–R laid alphabetically along a counter-clockwise spiral.",
};

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
    <div style={{ display: "flex", flexDirection: "column", gap: 4, margin: "2px 0 6px 12px" }}>
      {Object.entries(spec.params).map(([name, p]) => (
        <div key={name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ ...labelStyle, width: 200, textTransform: "none" }}>{name}</span>
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
  // The concrete map seed (always shown); the dice rerolls it, the input sets it.
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 65536));
  const [preview, setPreview] = useState<Board | null>(null);
  // With several human seats: all on this screen, or just yours (the others
  // join through the invite link).
  const [seating, setSeating] = useState<"hotseat" | "online">("hotseat");
  // One controller per possible seat; only the first nPlayers are used.
  const [seats, setSeats] = useState<SeatConfig[]>([
    { kind: HUMAN },
    { kind: "random" },
    { kind: "random" },
    { kind: "random" },
  ]);
  // The seat whose bot is being picked (the in-dialog bot page), or null.
  const [pickerSeat, setPickerSeat] = useState<number | null>(null);
  // Whether the in-dialog map picker page is open.
  const [mapOpen, setMapOpen] = useState(false);
  const [bots, setBots] = useState<Record<string, BotSpec>>({});

  useEffect(() => {
    fetchBots().then(setBots).catch(() => setBots({}));
  }, []);

  // Bot kinds available at the chosen player count; a seat holding a kind
  // that the new count doesn't support falls back to "random".
  const botNames = Object.keys(bots)
    .filter((b) => bots[b].counts.includes(nPlayers))
    .sort();
  const defaultBot = botNames.includes("random") ? "random" : (botNames[0] ?? "random");
  useEffect(() => {
    setSeats((prev) =>
      prev.map((s) =>
        s.kind === HUMAN || bots[s.kind]?.counts.includes(nPlayers) ? s : { kind: "random" }
      )
    );
  }, [nPlayers, bots]);

  // Escape backs out of a sub-page (bot / map) first, then closes the dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (pickerSeat !== null) setPickerSeat(null);
      else if (mapOpen) setMapOpen(false);
      else onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, pickerSeat, mapOpen]);

  // Keep a live board preview for the chosen seed / count / number placement.
  useEffect(() => {
    let cancelled = false;
    const t = setTimeout(() => {
      fetchPreview(seed, nPlayers, numberPlacement)
        .then((b) => !cancelled && setPreview(b))
        .catch(() => !cancelled && setPreview(null));
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [seed, nPlayers, numberPlacement]);

  // Just the ocean board (no bank / seats) for the picker.
  const previewBoard = useMemo<Board | null>(
    () => (preview ? { ...preview, bank: undefined, players: [] } : null),
    [preview]
  );

  const reroll = () => setSeed(Math.floor(Math.random() * 65536));

  // Switching the number layout rerolls the map too, so the change is visibly a
  // fresh board rather than the same tiles with shuffled tokens.
  const pickPlacement = (p: NumberPlacement) => {
    setNumberPlacement(p);
    reroll();
  };

  const humanSeats = seats.slice(0, nPlayers).filter((s) => s.kind === HUMAN).length;

  const setSeat = (i: number, seat: SeatConfig) =>
    setSeats((prev) => prev.map((p, j) => (j === i ? seat : p)));

  const openPicker = (i: number) => {
    if (seats[i].kind === HUMAN) setSeat(i, { kind: defaultBot });
    setPickerSeat(i);
  };

  const start = () => {
    onStart({
      seed,  // the seed the preview showed
      nPlayers,
      numberPlacement,
      seats: seats.slice(0, nPlayers),
      claim: humanSeats >= 2 && seating === "online" ? "first" : "all",
    });
  };

  const panelInner: React.CSSProperties = {
    ...panelStyle,
    display: "flex",
    flexDirection: "column",
    gap: 14,
    padding: "20px 24px",
    minWidth: 320,
    maxWidth: 380,
  };

  // The bot-picker page for one seat.
  if (pickerSeat !== null) {
    const seat = seats[pickerSeat];
    return (
      <Overlay onClose={onClose}>
        <div style={panelInner} onClick={(e) => e.stopPropagation()}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button style={{ ...buttonStyle, padding: "4px 10px", fontSize: 12 }} onClick={() => setPickerSeat(null)}>
              ‹ Back
            </button>
            <span style={{ fontSize: 16, fontWeight: 700, display: "inline-flex", alignItems: "center", gap: 6 }}>
              <BotIcon size={17} /> {playerName(pickerSeat)}'s bot
            </span>
          </div>
          {botNames.map((name) => {
            const spec = bots[name];
            const selected = seat.kind === name;
            return (
              <div key={name} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <button
                  onClick={() =>
                    setSeat(pickerSeat, { kind: name, params: seat.kind === name ? seat.params : {} })
                  }
                  style={{
                    ...buttonStyle,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    gap: 3,
                    padding: "8px 12px",
                    textAlign: "left",
                    width: "100%",
                    ...(selected ? selectedStyle : {}),
                  }}
                >
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{botLabel(name)}</span>
                  <span style={{ fontSize: 11, opacity: 0.75, fontWeight: 400 }}>{spec.description}</span>
                </button>
                {selected && Object.keys(spec.params).length > 0 && (
                  <SeatParams
                    spec={spec}
                    params={seat.params ?? {}}
                    onChange={(params) => setSeat(pickerSeat, { ...seat, params })}
                  />
                )}
              </div>
            );
          })}
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button style={{ ...buttonStyle, ...selectedStyle }} onClick={() => setPickerSeat(null)}>
              Done
            </button>
          </div>
        </div>
      </Overlay>
    );
  }

  // The map-picker page: number layout (with hover help), the seed (rerolled by
  // the dice or typed in), and the live preview.
  if (mapOpen) {
    return (
      <Overlay onClose={onClose}>
        <div style={panelInner} onClick={(e) => e.stopPropagation()}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button style={{ ...buttonStyle, padding: "4px 10px", fontSize: 12 }} onClick={() => setMapOpen(false)}>
              ‹ Back
            </button>
            <span style={{ fontSize: 16, fontWeight: 700 }}>Map</span>
          </div>
          <Toggle
            label="Numbers"
            options={["random", "spiral"] as const}
            value={numberPlacement}
            onChange={pickPlacement}
            optionTitle={(o) => PLACEMENT_HELP[o]}
          />
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={labelStyle}>Seed</span>
            <button
              style={{ ...buttonStyle, padding: "5px 10px", fontSize: 14 }}
              onClick={reroll}
              title="Roll a new random map"
            >
              🎲
            </button>
            <input
              type="number"
              value={seed}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (e.target.value !== "" && Number.isFinite(v)) setSeed(Math.max(0, Math.round(v)));
              }}
              title="The map seed — reroll for a new one, or type a specific seed"
              style={{ ...buttonStyle, cursor: "text", width: 110, padding: "5px 10px", fontSize: 12 }}
            />
          </div>
          <div
            style={{
              position: "relative",
              width: "100%",
              height: 240,
              borderRadius: 8,
              overflow: "hidden",
              background: "#0D3B66",
            }}
          >
            {previewBoard && <BoardView board={previewBoard} />}
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button style={{ ...buttonStyle, ...selectedStyle }} onClick={() => setMapOpen(false)}>
              Done
            </button>
          </div>
        </div>
      </Overlay>
    );
  }

  return (
    <Overlay onClose={onClose}>
      <div style={panelInner} onClick={(e) => e.stopPropagation()}>
        <span style={{ fontSize: 18, fontWeight: 700 }}>New game</span>
        <Toggle label="Players" options={[2, 4] as const} value={nPlayers} onChange={setNPlayers} />
        {seats.slice(0, nPlayers).map((seat, i) => {
          const isHuman = seat.kind === HUMAN;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={labelStyle}>{playerName(i)}</span>
              <button
                style={{ ...seatButtonStyle, ...(isHuman ? selectedStyle : {}) }}
                onClick={() => setSeat(i, { kind: HUMAN })}
              >
                <HumanIcon /> Human
              </button>
              <button
                title="Choose and configure a bot"
                style={{ ...seatButtonStyle, ...(isHuman ? {} : selectedStyle) }}
                onClick={() => openPicker(i)}
              >
                <BotIcon /> {isHuman ? "Bot" : botLabel(seat.kind)}
              </button>
            </div>
          );
        })}
        {humanSeats >= 2 && (
          <Toggle
            label="Seating"
            options={["hotseat", "online"] as const}
            value={seating}
            onChange={setSeating}
            trailing={
              <span style={{ fontSize: 11, opacity: 0.5 }}>
                {seating === "online" ? "others join via the invite link" : "all on this screen"}
              </span>
            }
          />
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={labelStyle}>Map</span>
          <button
            style={{
              ...buttonStyle,
              padding: "5px 12px",
              fontSize: 12,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
            onClick={() => setMapOpen(true)}
            title="Choose the map: seed and number layout"
          >
            <MapIcon /> <span style={{ opacity: 0.8 }}>{numberPlacement} · #{seed}</span>
          </button>
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
    </Overlay>
  );
}

// The shared modal backdrop; clicking it closes the dialog.
function Overlay({ onClose, children }: { onClose: () => void; children: React.ReactNode }) {
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
      {children}
    </div>
  );
}
