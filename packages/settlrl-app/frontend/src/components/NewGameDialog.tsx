// Modal dialog configuring a new game: players, per-seat Human/Bot choice,
// number placement, seed. Choosing Bot opens an in-dialog picker listing each
// available bot with its description (one bot service = one configured bot).

import { useEffect, useMemo, useState } from "react";
import {
  fetchBots,
  fetchPreview,
  HUMAN,
  type BotSpec,
  type NewGameConfig,
  type NumberPlacement,
  type PlayerCount,
  type SeatConfig,
} from "../lib/game";
import { authToken } from "../lib/auth";
import { type Board, playerName } from "../lib/boardData";
import { cx } from "../lib/cx";
import ui from "../styles/ui.module.css";
import BoardView from "./BoardView";
import Button from "./Button";
import Modal from "./Modal";
import { BotIcon, HumanIcon, MapIcon } from "./icons";
import s from "./NewGameDialog.module.css";

const botLabel = (kind: string, spec?: BotSpec) =>
  spec?.title ?? (kind === "mcts" ? "MCTS" : kind.charAt(0).toUpperCase() + kind.slice(1));

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
    <div className={s.row}>
      <span className={s.label}>{label}</span>
      {options.map((o) => (
        <button
          key={o}
          title={optionTitle?.(o)}
          className={cx(s.toggleButton, value === o && ui.selected)}
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

export default function NewGameDialog({
  onStart,
  onClose,
  defaultOnline = false,
}: {
  onStart: (config: NewGameConfig) => void;
  onClose: () => void;
  // Open as an online multiplayer game (two human seats, online seating) — the
  // lobby host flow, where the common case is others joining from their devices.
  defaultOnline?: boolean;
}) {
  const [nPlayers, setNPlayers] = useState<PlayerCount>(4);
  const [numberPlacement, setNumberPlacement] = useState<NumberPlacement>("random");
  // The concrete map seed (always shown); the dice rerolls it, the input sets it.
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 65536));
  const [preview, setPreview] = useState<Board | null>(null);
  // With several human seats: all on this screen, or just yours (the others
  // join through the invite link).
  const [seating, setSeating] = useState<"hotseat" | "online">(
    defaultOnline ? "online" : "hotseat"
  );
  // List the game in the public lobby so anyone can take its open human seats.
  // Listing requires a signed-in account (the server enforces it too).
  const signedIn = !!authToken();
  const [listed, setListed] = useState(defaultOnline && signedIn);
  // Mark the game open to Quick Match (a visibility flag shown in the lobby).
  const [searchable, setSearchable] = useState(false);
  // One controller per possible seat; only the first nPlayers are used. The
  // online host flow opens with two human seats so others have a seat to take.
  const [seats, setSeats] = useState<SeatConfig[]>([
    { kind: HUMAN },
    { kind: defaultOnline ? HUMAN : "random" },
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

  // Escape backs out of a sub-page (bot / map) first, then closes the dialog
  // (Radix closes it for us once there's no sub-page to back out of).
  const handleEscape = (e: KeyboardEvent) => {
    if (pickerSeat !== null) {
      e.preventDefault();
      setPickerSeat(null);
    } else if (mapOpen) {
      e.preventDefault();
      setMapOpen(false);
    }
  };

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
  // Online seating (others join remotely) is only meaningful with >=2 humans;
  // only then are seats left open, so only then can the game be listed.
  const online = humanSeats >= 2 && seating === "online";

  const setSeat = (i: number, seat: SeatConfig) =>
    setSeats((prev) => prev.map((p, j) => (j === i ? seat : p)));

  const openPicker = (i: number) => {
    if (seats[i].kind === HUMAN) setSeat(i, { kind: defaultBot });
    setPickerSeat(i);
  };

  const start = () => {
    onStart({
      seed, // the seed the preview showed
      nPlayers,
      numberPlacement,
      seats: seats.slice(0, nPlayers),
      claim: online ? "first" : "all",
      // Only an online game leaves human seats open for others to take, and
      // only a signed-in creator may list it publicly or open it to Quick Match.
      listed: online && signedIn && listed,
      searchable: online && signedIn && searchable,
    });
  };

  // The bot-picker page for one seat.
  if (pickerSeat !== null) {
    const seat = seats[pickerSeat];
    return (
      <Modal onClose={onClose} title={`${playerName(pickerSeat)}'s bot`} onEscapeKeyDown={handleEscape}>
        <div className={s.dialog}>
          <div className={s.pickerHeader}>
            <button className={s.backButton} onClick={() => setPickerSeat(null)}>
              ‹ Back
            </button>
            <span className={s.pickerTitle}>
              <BotIcon size={17} /> {playerName(pickerSeat)}'s bot
            </span>
          </div>
          {botNames.map((name) => {
            const spec = bots[name];
            const selected = seat.kind === name;
            return (
              <button
                key={name}
                onClick={() => setSeat(pickerSeat, { kind: name })}
                className={cx(s.botOption, selected && ui.selected)}
              >
                <span className={s.botName}>{botLabel(name, spec)}</span>
                <span className={s.botDesc}>{spec.description}</span>
              </button>
            );
          })}
          <div className={s.footerEnd}>
            <Button selected onClick={() => setPickerSeat(null)}>
              Done
            </Button>
          </div>
        </div>
      </Modal>
    );
  }

  // The map-picker page: number layout (with hover help), the seed (rerolled by
  // the dice or typed in), and the live preview.
  if (mapOpen) {
    return (
      <Modal onClose={onClose} title="Map" onEscapeKeyDown={handleEscape}>
        <div className={s.dialog}>
          <div className={s.pickerHeader}>
            <button className={s.backButton} onClick={() => setMapOpen(false)}>
              ‹ Back
            </button>
            <span className={s.pickerTitle}>Map</span>
          </div>
          <Toggle
            label="Numbers"
            options={["random", "spiral"] as const}
            value={numberPlacement}
            onChange={pickPlacement}
            optionTitle={(o) => PLACEMENT_HELP[o]}
          />
          <div className={s.row}>
            <span className={s.label}>Seed</span>
            <button className={s.diceButton} onClick={reroll} title="Roll a new random map">
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
              className={s.seedInput}
            />
          </div>
          <div className={s.preview}>{previewBoard && <BoardView board={previewBoard} />}</div>
          <div className={s.footerEnd}>
            <Button selected onClick={() => setMapOpen(false)}>
              Done
            </Button>
          </div>
        </div>
      </Modal>
    );
  }

  return (
    <Modal onClose={onClose} title="New game" onEscapeKeyDown={handleEscape}>
      <div className={s.dialog}>
        <span className={s.heading}>New game</span>
        <Toggle label="Players" options={[2, 4] as const} value={nPlayers} onChange={setNPlayers} />
        {seats.slice(0, nPlayers).map((seat, i) => {
          const isHuman = seat.kind === HUMAN;
          return (
            <div key={i} className={s.row}>
              <span className={s.label}>{playerName(i)}</span>
              <button
                className={cx(s.seatButton, isHuman && ui.selected)}
                onClick={() => setSeat(i, { kind: HUMAN })}
              >
                <HumanIcon /> Human
              </button>
              <button
                title="Choose and configure a bot"
                className={cx(s.seatButton, !isHuman && ui.selected)}
                onClick={() => openPicker(i)}
              >
                <BotIcon /> {isHuman ? "Bot" : botLabel(seat.kind, bots[seat.kind])}
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
              <span className={s.hint}>
                {seating === "online" ? "others join via the invite link" : "all on this screen"}
              </span>
            }
          />
        )}
        {online && (
          <div className={s.row}>
            <span className={s.label}>Lobby</span>
            <button
              className={cx(s.seatButton, listed && signedIn && ui.selected)}
              onClick={() => signedIn && setListed((v) => !v)}
              disabled={!signedIn}
              title={
                signedIn
                  ? "Show this game in the public lobby so anyone can join its open seats"
                  : "Sign in to list a game in the public lobby"
              }
            >
              {listed && signedIn ? "Listed" : "List in lobby"}
            </button>
            <span className={s.hint}>
              {!signedIn ? "sign in to list" : listed ? "anyone can join" : "invite link only"}
            </span>
          </div>
        )}
        {online && (
          <div className={s.row}>
            <span className={s.label}>Quick Match</span>
            <button
              className={cx(s.seatButton, searchable && signedIn && ui.selected)}
              onClick={() => signedIn && setSearchable((v) => !v)}
              disabled={!signedIn}
              title={
                signedIn
                  ? "Mark this game as open to Quick Match players"
                  : "Sign in to open a game to Quick Match"
              }
            >
              {searchable && signedIn ? "Searchable" : "Open to Quick Match"}
            </button>
            <span className={s.hint}>
              {!signedIn ? "sign in to enable" : searchable ? "matched players can find it" : "off"}
            </span>
          </div>
        )}
        <div className={s.row}>
          <span className={s.label}>Map</span>
          <button
            className={s.mapButton}
            onClick={() => setMapOpen(true)}
            title="Choose the map: seed and number layout"
          >
            <MapIcon /> <span className={s.dim}>{numberPlacement} · #{seed}</span>
          </button>
        </div>
        <div className={s.footer}>
          <Button onClick={onClose}>Cancel</Button>
          <Button selected onClick={start}>
            Start
          </Button>
        </div>
      </div>
    </Modal>
  );
}
