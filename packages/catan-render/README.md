# catan-render

Web-based renderer for catan-engine. FastAPI serves board state over a JSON API; a Vite + React + TypeScript frontend renders the board as SVG.

A menu lets you choose between two modes, each at its own URL:

- **Play** (`/play`) — a live, playable game. Each seat is configured per game as a human
  sharing the screen (hotseat) or a bot (a `catan-agents` policy, e.g. random or greedy —
  search bots expose their build parameters, like `mcts`'s simulation budget, behind the seat's gear button);
  with no human seats the game plays itself as a spectated bot match. Input follows the board:
  every placement you can currently make is marked on the board in your colour (hover to
  preview the piece) — click it and confirm in a popup there (which shows the build cost, or
  who to rob). The hand is live too: click a glowing development card to play it (resource
  picks come from a popover) and resource cards to discard after a 7. A small bottom bar keeps
  the turn-flow moves (roll, buy a dev card, bank trade, end turn); the top bar holds New
  game and a light/dark theme toggle. The scene is a top-down table:
  the bank's card piles (true to card scale) sit left of the board, and each seat's play
  area lines its table edge — face-down hand and dev piles plus their unbuilt roads,
  settlements, and cities, so supplies are read straight off the table. The bar
  also shows the acting human's hand (resources + dev cards by type). The right column opens
  with the seats in playing order (⭐ points, 🎴 cards, 🃏 devs — and 🔍 to inspect an
  opponent's proven hand bounds, i.e. public card counting) over the chat panel, which doubles
  as the game log: the server logs every move as it is played (and the win), and
  humans can post messages. A help page (`/help`, the **?** button top-left) documents the
  controls and icons. On entry — and from the
  **New game** button — a dialog configures the next game: player count (2 or 4), what controls
  each seat, number-token placement (random or spiral), and an optional seed;
  cancelling keeps the game in progress.
- **Replay** (`/replay`) — step through a recorded game. Load a saved game-record file (the
  JSON from `GET /api/game/record`) or the live game as played so far, then scrub anywhere
  with the slider, step move by move, or press play; the log panel fills in as the game
  advances, and the record can be saved back to a file. The server replays the record through
  the engine once and serves the board after every move.

The game is driven through `catan-engine`'s single-game PettingZoo-AEC env
(`catan_engine.env.aec`): the server holds one live game and applies your moves. Bot seats
are stepped one move per request (`POST /api/game/bot`), so the frontend paces them with a
short delay and animates each move (pieces pop in, the robber slides, and the bar shows what
each bot just played).

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18

## Development

Run the API server and the frontend dev server in separate terminals from the repo root.

**Terminal 1 — API (port 8000)**
```bash
uv run catan-render
```

The server runs JAX on CPU by default (one live game doesn't need a GPU, and JAX would
otherwise preallocate most of its memory). Set `JAX_PLATFORMS=cuda` to override.

**Terminal 2 — frontend (port 5173)**
```bash
cd packages/catan-render/frontend
npm install   # first time only
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the FastAPI server, so hot-reload and the API work together out of the box.

## Production build

Build the frontend into `frontend/dist/`, then start the API server — it detects the built assets and serves them automatically.

```bash
cd packages/catan-render/frontend && npm run build
uv run catan-render
```

Open http://localhost:8000.

## Tests

The renderer builds its board coordinate tables directly from the engine's
authoritative geometry lookups, and derives resource / dev-card orderings from
the engine enums, so those can't drift. It still mirrors the AEC flat action
table; the test suite pins that decode against the engine's own lookups, checks
the enum-derived orderings, and exercises the conversion and HTTP layers — if the
engine reindexes the board, reorders an enum, or changes the action table, these
tests fail.

```bash
uv run pytest packages/catan-render/tests
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/board` | Returns the current board as JSON |
| `GET /api/game` | Live game snapshot: board + turn status + your legal moves + the game log |
| `POST /api/game/action` | Apply your move `{ "flat": <action index> }`; returns the new snapshot. `409` if the move is illegal |
| `POST /api/game/bot` | Play one due bot move; the snapshot's `bot_move` says who played what (null when it's a human's turn) |
| `POST /api/game/chat` | Append a chat message to the game log `{ "text": <string>, "player": <seat> \| null }` |
| `GET /api/game/record` | Download the game as a `catan_engine.record` JSON transcript — self-contained and replayable (`winner` is null while the game is running) |
| `POST /api/replay` | Load a game record (the `GET /api/game/record` JSON) for replay; returns the opening state. `422` if it's malformed or fails replay validation |
| `POST /api/replay/from-game` | Load the live game (as played so far) for replay |
| `GET /api/replay/state?move=N` | The loaded replay after `N` moves (0 = the opening board): board + the moves played up to that point. `404` until a replay is loaded |
| `GET /api/replay/record` | The loaded replay's record JSON (to save it to a file) |
| `GET /api/bots` | Bot kinds available for seats, each with the player counts it supports and its configurable parameters (`{counts, params: {name: {type, default}}}`) |
| `POST /api/game/reset` | Start a fresh game `{ "seed": <int>, "n_players": 2 \| 4, "number_placement": "random" \| "spiral", "seats": [...] }` — each seat `"human"`, a bot kind, or a configured bot `{ "kind", "params" }` with knob overrides from `GET /api/bots` (one entry per seat; default: you + 3 random bots) |
| `GET /docs` | Interactive API docs (Swagger UI) |

Each legal move in `GET /api/game` is a decoded action descriptor carrying its `flat` index
(post it back to apply it), a `type`, a human `label`, and — depending on the type — the board
target (`vertex` / `edge` / `tile` + `victim`) or resources involved, in the same cube/axial
coordinates the board uses.

Example response:
```json
{
  "tiles": [
    { "q": 0, "r": -2, "terrain": "ore", "number": 10 },
    { "q": 1, "r": -2, "terrain": "sheep", "number": 2 },
    ...
  ]
}
```

Tile position uses **axial coordinates** with a pointy-top hex orientation. The board is a hexagon of radius 2 (19 tiles) centred on `(0, 0)`. The layout (terrain and number tokens) is generated by `catan-engine`, so it is randomised per server start rather than fixed.

## Project layout

```
packages/catan-render/
├── src/catan_render/
│   ├── __init__.py      # CLI entry point (uvicorn)
│   ├── server.py        # FastAPI app: /api/board + /api/game* + /api/replay* endpoints
│   ├── session.py       # GameSession: one live game vs bots (wraps the AEC env)
│   ├── replay.py        # ReplaySession: a loaded record replayed into per-move snapshots
│   ├── bots.py          # catan-agents registry adapted to the single game (bot_act)
│   ├── actions.py       # Decode AEC flat actions -> wire ActionModels
│   ├── convert.py       # catan-engine Board -> BoardModel
│   └── models.py        # Pydantic board / game / action models
├── tests/               # Pytest: renderer<->engine sync checks (geometry, actions, enums)
└── frontend/
    └── src/
        ├── App.tsx          # Routes: menu, /play, /help, /replay
        ├── lib/hex.ts        # Axial/cube → pixel conversion, hex corner math, coord equality
        ├── lib/api.ts        # JSON fetch wrapper (ApiError)
        ├── lib/boardData.ts  # Board types + palette + adaptBoard (wire -> camelCase)
        ├── lib/game.ts       # Live-game API client (/api/game*)
        ├── lib/replay.ts     # Replay API client (/api/replay*)
        ├── lib/actionMeta.ts # Icon + label per action type (control bar + help page)
        ├── lib/useGame.ts    # Hook driving the live game (act / reset)
        ├── lib/ui.ts         # Shared panel / button / highlight styles
        ├── pages/
        │   ├── Menu.tsx       # Landing page: choose Play or Replay
        │   ├── PlayView.tsx   # Play mode: interactive board + live action bar
        │   ├── HelpView.tsx   # Help page: controls, action icons, seats
        │   └── ReplayView.tsx # Replay mode: load a record, scrub / step / play it
        └── components/
            ├── TopBar.tsx       # Back-to-menu + mode label bar
            ├── BoardView.tsx    # SVG viewport, zoom + pan, optional click/highlight interaction
            ├── NewGameDialog.tsx # Modal: configure players / seats / numbers / seed for a new game
            ├── ChatPanel.tsx    # Right-hand chat / log column (Play; read-only in Replay)
            ├── HexTile.tsx      # Hex polygon, terrain colour + motifs, number token
            ├── TerrainIcon.tsx  # Per-terrain silhouette motif (pine, sheep, …)
            ├── Road.tsx         # Player road along an edge
            ├── Building.tsx     # Settlement / city on a vertex
            ├── Robber.tsx       # Robber pawn on a tile
            ├── Port.tsx         # Harbour badge (2:1 / 3:1) with docks
            └── PlayerPanel.tsx  # Per-player corner panel (cards / dev / VP)
```
