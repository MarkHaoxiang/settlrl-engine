# catan-engine library

## Source files

Keep this section up to date when adding, removing, or significantly changing a file.

- `tile.py` — `Tile` enum: the six resource types (sheep, wheat, wood, brick, ore, desert).
- `port.py` — `Port` enum: the six port types (one per resource plus the 3:1 general port).
- `board.py` — Static board representation. Defines `BoardStatic` (tile resources, number tokens, port allocation), the module-level jaxtyping aliases for its arrays, the tile-vertex and port-vertex mappings, and `make_board()` which randomly generates a batched `BoardStatic`.
- `state.py` — Mutable game state. Defines `BoardState` (vertex ownership/type, edge roads, robber, player resources, victory points), jaxtyping aliases for its arrays, and `make_board_state()` which initialises an empty `BoardState`.
- `game.py` — Top-level game types. Defines `Board = tuple[BoardStatic, BoardState]`.
- `resources.py` — Resource constants (`N_PLAYERS`, `N_RESOURCES`, `BANK_INITIAL`), jaxtyping aliases for resource arrays, and `compute_bank_resources()` which derives remaining bank stock from player holdings.
- `env.py` — (placeholder) RL environment entry point.
- `__init__.py` — Package entry point (`main`).
