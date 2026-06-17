// Action sound effects: short CC0 clips (Kenney) played off the game's log.
//
// Sounds are derived from new log entries (see useGame), so they cover human
// and bot moves alike with no extra wire data. Muted by default — the toggle
// (TopBar) persists in localStorage. Audio is unlocked on the first user
// gesture (browsers block it until then) and decoded lazily via Web Audio, so
// rapid events can overlap without re-fetching.

import type { LogEntry } from "./game";

const FILES = {
  roll: "roll.ogg",
  build: "build.ogg",
  card: "card.ogg",
  robber: "robber.ogg",
  trade: "trade.ogg",
  win: "win.ogg",
  chat: "chat.ogg",
  error: "error.ogg",
} as const;

export type SoundName = keyof typeof FILES;

const MUTE_KEY = "settlrl.sound.muted";
// Default muted: only unmuted once the player has explicitly turned sound on.
let muted = localStorage.getItem(MUTE_KEY) !== "false";

let ctx: AudioContext | null = null;
const buffers = new Map<SoundName, AudioBuffer>();
const loading = new Map<SoundName, Promise<AudioBuffer | null>>();

function context(): AudioContext {
  ctx ??= new AudioContext();
  return ctx;
}

async function buffer(name: SoundName): Promise<AudioBuffer | null> {
  const ready = buffers.get(name);
  if (ready) return ready;
  let pending = loading.get(name);
  if (!pending) {
    pending = (async () => {
      try {
        const res = await fetch(`/sounds/${FILES[name]}`);
        const decoded = await context().decodeAudioData(await res.arrayBuffer());
        buffers.set(name, decoded);
        return decoded;
      } catch {
        return null; // missing/undecodable clip: stay silent rather than throw
      }
    })();
    loading.set(name, pending);
  }
  return pending;
}

export function isMuted(): boolean {
  return muted;
}

export function setMuted(value: boolean): void {
  muted = value;
  localStorage.setItem(MUTE_KEY, String(value));
  if (!value) void context().resume(); // turning on counts as the unlock gesture
}

export async function play(name: SoundName): Promise<void> {
  if (muted) return;
  const c = context();
  if (c.state === "suspended") await c.resume();
  const buf = await buffer(name);
  if (!buf || muted) return; // re-check: mute may have flipped while decoding
  const source = c.createBufferSource();
  source.buffer = buf;
  const gain = c.createGain();
  gain.gain.value = 0.5;
  source.connect(gain).connect(c.destination);
  source.start();
}

// The sound a freshly-applied log entry should make, or null for silent ones
// (end turn, discard, …). Robber is matched before the generic dev-card play.
export function soundForLog(entry: LogEntry): SoundName | null {
  if (entry.kind === "win") return "win";
  if (entry.kind === "chat") return "chat";
  if (entry.kind !== "move") return null;
  const type = entry.action_type ?? "";
  if (type === "move_robber" || type === "play_knight") return "robber";
  if (type === "roll_dice") return "roll";
  if (type === "buy_development_card" || type.startsWith("play_")) return "card";
  if (type.startsWith("build") || type.startsWith("setup")) return "build";
  if (type.includes("trade")) return "trade";
  return null;
}

// Resume the audio context on the first user gesture so the first real sound
// isn't swallowed by the browser's autoplay policy.
if (typeof window !== "undefined") {
  const unlock = () => {
    void context().resume();
    window.removeEventListener("pointerdown", unlock);
  };
  window.addEventListener("pointerdown", unlock);
}
