# catan-render

Web-based renderer for catan-engine. FastAPI serves board state over a JSON API; a Vite + React + TypeScript frontend renders the board as SVG.

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18

## Development

Run the API server and the frontend dev server in separate terminals from the repo root.

**Terminal 1 — API (port 8000)**
```bash
uv run catan-render
```

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

## API

| Endpoint | Description |
|---|---|
| `GET /api/board` | Returns the current board as JSON |
| `GET /docs` | Interactive API docs (Swagger UI) |

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

Tile position uses **axial coordinates** with a pointy-top hex orientation. The board is a hexagon of radius 2 (19 tiles), with the desert at `(0, 0)`.

## Project layout

```
packages/catan-render/
├── src/catan_render/
│   ├── __init__.py      # CLI entry point (uvicorn)
│   ├── server.py        # FastAPI app and /api/board
│   └── models.py        # Pydantic board / tile models
└── frontend/
    └── src/
        ├── lib/hex.ts        # Axial → pixel conversion, hex corner math
        ├── lib/boardData.ts  # Hardcoded standard Catan board (v1)
        └── components/
            ├── CatanBoard.tsx  # SVG viewport and ocean background
            └── HexTile.tsx     # Hex polygon, terrain colour, number token
```
