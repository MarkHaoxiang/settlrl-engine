from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .models import BoardModel, Terrain, TileModel

app = FastAPI(title="Catan Render")

# Standard Catan board — desert at centre (0,0)
_BOARD = BoardModel(
    tiles=[
        # Row r=-2 (top, 3 tiles)
        TileModel(q=0, r=-2, terrain=Terrain.ore, number=10),
        TileModel(q=1, r=-2, terrain=Terrain.sheep, number=2),
        TileModel(q=2, r=-2, terrain=Terrain.wood, number=9),
        # Row r=-1 (4 tiles)
        TileModel(q=-1, r=-1, terrain=Terrain.wheat, number=12),
        TileModel(q=0, r=-1, terrain=Terrain.brick, number=6),
        TileModel(q=1, r=-1, terrain=Terrain.sheep, number=4),
        TileModel(q=2, r=-1, terrain=Terrain.brick, number=10),
        # Row r=0 (5 tiles, middle)
        TileModel(q=-2, r=0, terrain=Terrain.wood, number=9),
        TileModel(q=-1, r=0, terrain=Terrain.wheat, number=11),
        TileModel(q=0, r=0, terrain=Terrain.desert),
        TileModel(q=1, r=0, terrain=Terrain.wheat, number=3),
        TileModel(q=2, r=0, terrain=Terrain.ore, number=8),
        # Row r=1 (4 tiles)
        TileModel(q=-2, r=1, terrain=Terrain.sheep, number=8),
        TileModel(q=-1, r=1, terrain=Terrain.ore, number=3),
        TileModel(q=0, r=1, terrain=Terrain.brick, number=5),
        TileModel(q=1, r=1, terrain=Terrain.wheat, number=4),
        # Row r=2 (bottom, 3 tiles)
        TileModel(q=-2, r=2, terrain=Terrain.wood, number=5),
        TileModel(q=-1, r=2, terrain=Terrain.sheep, number=6),
        TileModel(q=0, r=2, terrain=Terrain.wood, number=11),
    ]
)


@app.get("/api/board")
def get_board() -> BoardModel:
    return _BOARD


# Serve built frontend when it exists
_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
