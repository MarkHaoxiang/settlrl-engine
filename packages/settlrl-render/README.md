# settlrl-render

Web-based renderer for settlrl-engine. FastAPI serves board state over a JSON API; a Vite + React + TypeScript frontend renders the board as SVG.

A menu lets you choose between two modes, each at its own URL:

- **Play** (`/play`) — a live, playable game. Each seat is configured per game as a human
  sharing the screen (hotseat) or a bot (a `settlrl-agents` policy, e.g. random or greedy —
  search bots expose their build parameters, like `mcts`'s simulation budget, behind the seat's gear button);
  with no human seats the game plays itself as a spectated bot match. Input follows the board:
  every placement you can currently make is marked on the board in your colour (hover to
  preview the piece) — click it and confirm in a popup there (which shows the build cost, or
  who to rob). The hand is live too: click a glowing development card to play it (resource
  picks come from a popover) and resource cards to discard after a 7 — a knight can also be
  played straight off the robber pawn, which lights up to take the click. Trading happens on
  the table as well: click the bank pile of the resource you want and a picker shows what to
  give and how many (your best port rate); click an opponent's hand pile to compose a 1:1
  offer (they accept or reject on their turn, the cards then crossing the table); and buy a
  development card by clicking the bank's deck. A small bottom bar keeps the rest (answer a
  trade, end turn — the dice rest in the table
  corner beside whoever's turn it is and take the click to roll); the top bar holds New
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
  each seat, seating with several humans (hotseat on this screen, or online — you take the first
  seat and the others join through the invite link), number-token placement (random or spiral),
  and an optional seed; cancelling keeps the game in progress.
- **Replay** (`/replay`) — step through a recorded game. Load a saved game-record file (the
  JSON from `GET /api/game/record`) or the live game as played so far, then scrub anywhere
  with the slider, step move by move, or press play; the log panel fills in as the game
  advances, and the record can be saved back to a file. The server replays the record through
  the engine once and serves the board after every move.

Each game is driven through `settlrl-engine`'s single-game PettingZoo-AEC env
(`settlrl_engine.env.aec`); the server holds many live games at once, addressed by id.
Claiming a human seat (creating or joining a game) issues a bearer token, and every request
proves its seats via the `X-Seat-Tokens` header: snapshots are per-seat views — your own
hand arrives in full, everyone else's only as public counts, and the legal-move list only
ships to the seat whose turn it is. Games are shareable: the 🔗 button copies the invite
link, and opening it claims a free human seat (or spectates when none is left); the 🔑 button
copies a resume link that carries your seat tokens, so you can restore the exact seats you
hold on another device or after clearing storage. The server
pushes state: each client holds an event stream (`GET /api/games/{id}/events`, SSE) and
receives its per-seat snapshot on every change, and bot seats are played by a server-side
driver pacing one move at a time so each lands as its own pushed snapshot — games advance
with no tab open, and every move animates.

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18

## Development

Run the API server and the frontend dev server in separate terminals from the repo root.

**Terminal 1 — API (port 8000)**
```bash
uv run settlrl-render
```

The server runs JAX on CPU by default (one live game doesn't need a GPU, and JAX would
otherwise preallocate most of its memory). Set `JAX_PLATFORMS=cuda` to override.

**Terminal 2 — frontend (port 5173)**
```bash
cd packages/settlrl-render/frontend
npm install   # first time only
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the FastAPI server, so hot-reload and the API work together out of the box.

## Production build

Build the frontend into `frontend/dist/`, then start the API server — it detects the built assets and serves them automatically.

```bash
cd packages/settlrl-render/frontend && npm run build
uv run settlrl-render
```

Open http://localhost:8000.

## Hosting

The server is configured by environment variables: `HOST` (default `0.0.0.0`),
`PORT` (default `8000`), `RELOAD` (default `1`; set `0` in production — the
reloader is a dev file-watcher),
`SETTLRL_RENDER_STATE_DIR` (a directory to persist games in — see below),
`SETTLRL_RENDER_TURN_TIMEOUT_S` (default `0` = off; after this many seconds of an
idle human turn the server auto-plays a move, so an abandoned game still
finishes instead of stalling), `SETTLRL_RENDER_MAX_ACTIVE` (default `16`; games
running at once before new creators are queued — keep it below the registry
cap), and `ROOT_PATH` (the proxy prefix when served under a path). Run **one
process, one worker**: live games are held in memory,
so extra workers would split them.
The registry holds up to 32 games, evicting finished games, hour-idle ones, or
unstarted ones idle past a few minutes (so a burst of empty games can't pin
every slot) to make room.

**Persistence.** Without `SETTLRL_RENDER_STATE_DIR`, games live only in memory and
a restart loses them. Point it at a (mounted) directory and each game is
journalled — its setup plus every move, seat claim, and chat line — into the same
SQLite database as accounts (`settlrl.db` there) and replayed back into the
registry on the next startup, so a deploy or crash resumes games in progress,
seat tokens and all. Bot pacing restarts for resumed games. Evicted games are
dropped from the database.

Anyone can create games; the concurrency cap queues them past
`SETTLRL_RENDER_MAX_ACTIVE`. For a public deployment, front the server with a
proxy that rate-limits — the built-in caps (a 2 MB request-body limit, a replay
move-count cap, and high-entropy game ids) bound resource use but are not a
substitute for one.

The repo-root `Dockerfile` builds a self-contained image (frontend compiled
in, CPU JAX):

```bash
docker build -t settlrl-render .
docker run -p 8000:8000 settlrl-render
```

To serve under a path instead of a (sub)domain — e.g. `markhaoxiang.com/settlrl`
behind a proxy that strips the prefix (Caddy `handle_path /settlrl/*`) — bake
the prefix into the frontend and tell FastAPI about it:

```bash
docker build -t settlrl-render --build-arg BASE_PATH=/settlrl/ .
docker run -p 8000:8000 -e ROOT_PATH=/settlrl settlrl-render
```

The mark-haoxiang repo's `infra/` wires this up as the `settlrl` compose
service behind its Caddy.

Seat tokens are bearer secrets, so put TLS in front for anything beyond a
LAN — e.g. Caddy, which manages certificates itself:

```
games.example.com {
    reverse_proxy localhost:8000
}
```

## Accounts

Accounts are optional: anonymous play (claim a seat, get a per-seat token) works
exactly as before. Registering gives a player a persistent identity and lets an
operator mark some users as **admins**, who manage the bot services below.

Accounts are handled by [fastapi-users](https://fastapi-users.github.io/fastapi-users/).
Login uses the OAuth2 password flow (`POST /api/auth/login` returns a bearer
token presented as `Authorization: Bearer …`); tokens are stored server-side, so
`POST /api/auth/logout` truly revokes one. Accounts, tokens, and games all share
the one SQLite database (`settlrl.db` under the state dir, or
`SETTLRL_RENDER_USER_DB`). Emails listed in `SETTLRL_RENDER_ADMIN_EMAILS`
(comma-separated) are granted admin on register and login. Endpoints:
`POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`,
`GET /api/users/me`.

**Seats follow the account.** A seat claimed while signed in is tied to your
user id, not just the per-device seat token, so you are recognised — and can
resume your games (`GET /api/me/games`) — on any device without carrying the
token. Send the bearer token alongside (or instead of) `X-Seat-Tokens`; each
snapshot's `your_seats` lists the seats the requester owns either way.

## Bot services

Where a seat's bot moves are computed is pluggable. By default the built-in
`settlrl-agents` policies run **in-process** (no change to how games are
served). They can instead — or additionally — run in a separate **bot service**,
so the agent compute is deployed and scaled apart from the game server.

A bot service is a small, stateless HTTP app (`settlrl-render-bot`) speaking a
standardized two-call API:

- `GET /catalog` — the bot kinds it offers (same shape as `GET /api/bots`).
- `POST /act` — given a game's setup and its flat moves so far (the data a
  `settlrl_engine.record` carries), it replays them and returns the acting
  seat's chosen flat action. No engine observation crosses the wire, so the two
  sides only agree on the (stable) record format and flat action indexing.

```bash
BOT_PORT=8100 uv run settlrl-render-bot
```

An **admin** registers a service at runtime; its kinds join the catalog and can
be seated like any built-in:

| Endpoint | Description |
|---|---|
| `GET /api/admin/bot-providers` | List registered remote bot services (admin) |
| `POST /api/admin/bot-providers` | Register one `{ "name", "base_url" }` (admin); `400` if unreachable or a kind clashes |
| `DELETE /api/admin/bot-providers/{name}` | Unregister one (admin) |

Set `SETTLRL_RENDER_LOCAL_BOTS=0` to run the game server with **no** in-process
agent execution at all: it then offers only registered services' kinds, so every
bot move is delegated over the API (an abandoned-turn auto-play still uses a
trivial local random move as a liveness fallback). A remote service that is slow
or fails simply falls back to a local random move, so a game never stalls.
Registrations live in memory, so re-register services after a restart.

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
uv run pytest packages/settlrl-render/tests
```

A browser end-to-end suite drives the real app (create / join / spectate and
per-seat redaction over the wire); it needs a running server with a built
frontend and a system Chromium:

```bash
cd packages/settlrl-render/frontend
BASE=http://localhost:8000 npm run e2e
```

## API

| Endpoint | Description |
|---|---|
| `POST /api/games` | Create a game `{ "seed", "n_players": 2 \| 4, "number_placement", "seats": [...], "claim": "all" \| "first" \| "none", "ticket"? }` — returns the game id and the creator's seat tokens. At the concurrency cap, returns `202` with a queue position `{ "queued": true, "ticket", "position", "total" }`; re-POST with the `ticket` to keep your place until a slot frees |
| `POST /api/games/{id}/join` | Claim a human seat `{ "seat"?: <n> }` (first free one by default) — returns the seat and its token. `409` when taken/full |
| `GET /api/games/{id}` | The requester's snapshot: board + status + their legal moves (`X-Seat-Tokens` header; omit to spectate) |
| `POST /api/games/{id}/action` | Apply the acting seat's move `{ "flat": <action index> }` — `403` without that seat's token, `409` if illegal |
| `GET /api/games/{id}/events` | Server-sent events: the requester's snapshot immediately, then again on every change (`bot_move` carries the server-paced bot play just made) |
| `POST /api/games/{id}/chat` | Append a chat message `{ "text", "player"?: <owned seat> }` (no seat: spectator) |
| `GET /api/games/{id}/record` | The finished game as a replayable `settlrl_engine.record` transcript (`409` while running: a record reconstructs hidden hands) |
| `POST /api/games/{id}/replay` | Load a finished game for replay (`409` while running) |
| `POST /api/replay` | Load a game record (the record JSON) for replay; returns the opening state. `422` if malformed |
| `GET /api/replay/state?move=N` | The loaded replay after `N` moves (0 = the opening board). `404` until a replay is loaded |
| `GET /api/replay/record` | The loaded replay's record JSON (to save it to a file) |
| `GET /api/bots` | Bot kinds available for seats (built-in + registered remote services), each with the player counts it supports and its configurable parameters |
| `POST /api/auth/register` · `/login` · `/logout` · `GET /api/users/me` | Optional accounts (OAuth2 password flow; see [Accounts](#accounts)) |
| `GET /api/me/games` | The signed-in user's live games — seats follow the account across devices |
| `GET` · `POST` · `DELETE /api/admin/bot-providers` | Manage remote bot services (admin; see [Bot services](#bot-services)) |
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

Tile position uses **axial coordinates** with a pointy-top hex orientation. The board is a hexagon of radius 2 (19 tiles) centred on `(0, 0)`. The layout (terrain and number tokens) is generated by `settlrl-engine`, so it is randomised per server start rather than fixed.

## Project layout

```
packages/settlrl-render/
├── src/settlrl_render/      # grouped by layer; server.py wires them together
│   ├── __init__.py      # CLI entry point (uvicorn)
│   ├── server.py        # create_app composition root: wires the app, mounts routers + SPA
│   ├── api/             # the HTTP layer
│   │   ├── deps.py        # Shared request helpers + the runtime context (Deps) routers close over
│   │   ├── routers/       # Routes by area: games, replay, bots, me (each build(deps) -> APIRouter)
│   │   ├── models.py      # Pydantic board / game / action wire models
│   │   ├── views.py       # Per-seat snapshots: the hidden-information boundary
│   │   ├── actions.py     # Decode AEC flat actions -> wire ActionModels
│   │   ├── convert.py     # settlrl-engine Board -> BoardModel
│   │   └── openapi.py     # Schema dump backing the generated frontend types
│   ├── game/            # the live game and its engine seam
│   │   ├── session.py     # GameSession: one live game vs bots (wraps the AEC env)
│   │   ├── games.py       # Game registry: ids, per-game locks, seat claims (tokens)
│   │   ├── driver.py      # Per-game asyncio task: bot pacing (local or remote) + idle-turn timeouts
│   │   └── replay.py      # ReplaySession: a loaded record replayed into per-move snapshots
│   ├── bots/            # where bot moves are computed
│   │   ├── bots.py        # settlrl-agents registry adapted to the single game (bot_act)
│   │   ├── providers.py   # Bot kinds -> where they run: local vs registered remote services
│   │   └── bot_service.py # Standalone bot service (settlrl-render-bot): /catalog + /act
│   └── storage/         # the one async DB: identity + persistence
│       ├── db.py          # The async SQLAlchemy engine: users, login tokens, and game journals
│       ├── auth.py        # Optional accounts: fastapi-users (DatabaseStrategy) on the shared DB
│       └── store.py       # Crash-recovery journals on the shared DB (write-behind), replay on boot
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
        ├── lib/transfers.ts  # Diff two snapshots into card-transfer animations (production / steals)
        ├── lib/replay.ts     # Replay API client (/api/replay*)
        ├── lib/actionMeta.ts # Action display metadata: icons, labels, costs, confirm phrasing
        ├── lib/useGame.ts    # Hook driving one live game (snapshot stream, act / chat)
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
            ├── ChoicePopover.tsx # Bottom-panel resource picker (monopoly / plenty)
            ├── MaritimePopover.tsx # Bank-pile picker: which resource to give + the rate
            ├── Hand.tsx         # The acting human's chips (resources, dev cards; clickable)
            ├── CountBadge.tsx   # Cream count badge for chip corners (matches CardPile's token)
            ├── CardPile.tsx     # Top-down card pile + count token (bank, player decks)
            ├── BankStacks.tsx   # The bank's card grid (resource piles + dev deck)
            ├── PlayerAreas.tsx  # Each seat's table edge: hand/dev piles + unbuilt pieces
            ├── TableDice.tsx    # The dice on the table (click to roll when glowing)
            ├── TransferAnimations.tsx # Chips that fly between bank piles / seats on a transfer
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
