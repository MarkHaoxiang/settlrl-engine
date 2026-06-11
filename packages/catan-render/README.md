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
  picks come from a popover) and resource cards to discard after a 7. Trading happens on the
  table as well: click the bank pile you want for the maritime exchanges, or an opponent's
  hand pile to compose a 1:1 offer (they accept or reject on their turn). A small bottom bar
  keeps the rest (buy a dev card, answer a trade, end turn — the dice sit on the table by
  the board's corner and take the click to roll); the top bar holds New
  game and a light/dark theme toggle. The scene is a top-down table (zoom, pan, and spin it — mouse, touch, or keyboard:
  arrows pan, +/− zoom, [ ] spin, 0 re-fits):
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

Each game is driven through `catan-engine`'s single-game PettingZoo-AEC env
(`catan_engine.env.aec`); the server holds many live games at once, addressed by id.
Claiming a human seat (creating or joining a game) issues a bearer token, and every request
proves its seats via the `X-Seat-Tokens` header: snapshots are per-seat views — your own
hand arrives in full, everyone else's only as public counts, and the legal-move list only
ships to the seat whose turn it is. Games are shareable: the 🔗 button copies the invite
link, and opening it claims a free human seat (or spectates when none is left). Bot seats
are stepped one move per request (`POST /api/games/{id}/bot`), so clients pace them with a
short delay and animate each move.

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
the enum-derived orderings, and exercises the conversion layer — if the engine
reindexes the board, reorders an enum, or changes the action table, these tests
fail.

The server tests follow its layering: `test_games.py` covers the registry and
seat claims without the engine, `test_views.py` covers the per-seat snapshots —
including a sweep asserting that no observer's view ever leaks another hand —
and `test_server.py` covers only what routes own (auth, status codes, locking),
each test building its own app via `create_app`. The wire contract is pinned
twice: pytest checks the committed `frontend/openapi.json` against the live
schema, and the frontend's wire types are generated from it (`npm run gen-api`
regenerates both whenever `models.py` changes).

```bash
uv run pytest packages/catan-render/tests
```

A browser end-to-end suite drives the real app (create / join / spectate and
per-seat redaction over the wire); it needs a running server with a built
frontend and a system Chromium:

```bash
cd packages/catan-render/frontend
BASE=http://localhost:8000 npm run e2e
```

## API

| Endpoint | Description |
|---|---|
| `POST /api/games` | Create a game `{ "seed", "n_players": 2 \| 4, "number_placement", "seats": [...], "claim": "all" \| "none" }` — returns the game id and the creator's seat tokens |
| `POST /api/games/{id}/join` | Claim a human seat `{ "seat"?: <n> }` (first free one by default) — returns the seat and its token. `409` when taken/full |
| `GET /api/games/{id}` | The requester's snapshot: board + status + their legal moves (`X-Seat-Tokens` header; omit to spectate) |
| `POST /api/games/{id}/action` | Apply the acting seat's move `{ "flat": <action index> }` — `403` without that seat's token, `409` if illegal |
| `POST /api/games/{id}/bot` | Play one due bot move; the snapshot's `bot_move` says who played what (null when none was due) |
| `POST /api/games/{id}/chat` | Append a chat message `{ "text", "player"?: <owned seat> }` (no seat: spectator) |
| `GET /api/games/{id}/record` | The finished game as a replayable `catan_engine.record` transcript (`409` while running: a record reconstructs hidden hands) |
| `POST /api/games/{id}/replay` | Load a finished game for replay (`409` while running) |
| `POST /api/replay` | Load a game record (the record JSON) for replay; returns the opening state. `422` if malformed |
| `GET /api/replay/state?move=N` | The loaded replay after `N` moves (0 = the opening board). `404` until a replay is loaded |
| `GET /api/replay/record` | The loaded replay's record JSON (to save it to a file) |
| `GET /api/bots` | Bot kinds available for seats, each with the player counts it supports and its configurable parameters |
| `GET /docs` | Interactive API docs (Swagger UI) |

Each legal move in `GET /api/games/{id}` is a decoded action descriptor carrying its `flat` index
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
│   ├── server.py        # create_app + thin routes: auth, locking, status codes
│   ├── views.py         # Per-seat snapshots: the hidden-information boundary
│   ├── games.py         # Game registry: ids, per-game locks, seat claims (tokens)
│   ├── openapi.py       # Schema dump backing the generated frontend types
│   ├── session.py       # GameSession: one live game vs bots (wraps the AEC env)
│   ├── replay.py        # ReplaySession: a loaded record replayed into per-move snapshots
│   ├── bots.py          # catan-agents registry adapted to the single game (bot_act)
│   ├── actions.py       # Decode AEC flat actions -> wire ActionModels
│   ├── convert.py       # catan-engine Board -> BoardModel
│   └── models.py        # Pydantic board / game / action models
├── tests/               # Pytest: renderer<->engine sync checks (geometry, actions, enums)
└── frontend/
    ├── openapi.json     # Committed wire schema (pinned by pytest; npm run gen-api)
    ├── e2e/             # Browser end-to-end checks (npm run e2e)
    └── src/
        ├── App.tsx          # Routes: menu, /play, /help, /replay
        ├── lib/hex.ts        # Axial/cube → pixel conversion, hex corner math, coord equality
        ├── lib/api.ts        # JSON fetch wrapper (ApiError)
        ├── lib/boardData.ts  # Board types + palette + resource/card constants + adaptBoard
        ├── lib/api-schema.d.ts # Wire types generated from openapi.json (do not edit)
        ├── lib/game.ts       # Live-game API client (/api/game*)
        ├── lib/replay.ts     # Replay API client (/api/replay*)
        ├── lib/actionMeta.ts # Action display metadata: icons, labels, costs, confirm phrasing
        ├── lib/useGame.ts    # Hook driving one live game (act / chat / bot pacing)
        ├── lib/seats.ts      # Seat tokens this browser holds, per game (localStorage)
        ├── lib/viewport.ts   # useTableViewport: pan / zoom / rotate (mouse, touch, keyboard)
        ├── lib/theme.ts      # Light / dark theme switching (persisted)
        ├── lib/ui.ts         # Shared panel / button / highlight styles (theme variables)
        ├── pages/
        │   ├── Menu.tsx       # Landing page: choose Play or Replay
        │   ├── PlayView.tsx   # Play mode: game state + handlers wiring the components below
        │   ├── HelpView.tsx   # Help page: controls, action icons, seats
        │   └── ReplayView.tsx # Replay mode: load a record, scrub / step / play it
        └── components/
            ├── TopBar.tsx       # Back-to-menu + mode label + theme toggle + view actions
            ├── BoardView.tsx    # The table scene: composes everything below in one SVG
            ├── InteractionOverlay.tsx # Legal-placement markers / hover ghosts / robber tiles
            ├── BoardPopover.tsx # Anchored action chooser (confirm + cost / victim pick)
            ├── ChoicePopover.tsx # Bottom-panel resource picker (monopoly / plenty / trades)
            ├── Hand.tsx         # The acting human's chips (resources, dev cards; clickable)
            ├── CardPile.tsx     # Top-down card pile + count token (bank, player decks)
            ├── BankStacks.tsx   # The bank's card grid (resource piles + dev deck)
            ├── PlayerAreas.tsx  # Each seat's table edge: hand/dev piles + unbuilt pieces
            ├── TableDice.tsx    # The dice on the table (click to roll when glowing)
            ├── PlayersPanel.tsx # Seat list atop the chat column (stats + belief inspect)
            ├── NewGameDialog.tsx # Modal: configure players / seats / numbers / seed for a new game
            ├── ChatPanel.tsx    # Right-hand column: players section + chat / log
            ├── ThemeToggle.tsx  # Light / dark switch
            ├── HexTile.tsx      # Hex polygon, terrain colour, icon-and-number token
            ├── TerrainIcon.tsx  # Per-terrain silhouette motif (pine, sheep, …)
            ├── Road.tsx         # Player road along an edge
            ├── Building.tsx     # Settlement / city on a vertex
            ├── Robber.tsx       # Robber pawn on a tile
            └── Port.tsx         # Harbour badge (2:1 / 3:1) with docks
```
