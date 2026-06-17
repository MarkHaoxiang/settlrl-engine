import os

import uvicorn


def main() -> None:
    # The server drives one game at a time; default JAX to CPU so importing it
    # doesn't preallocate most of the GPU's memory. Exported (not just set for
    # this process) so uvicorn's reloader subprocess inherits it; an explicit
    # JAX_PLATFORMS in the environment still wins.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Defaults suit development; deployments override via env (RELOAD=0 in
    # production — the reloader is a dev file-watcher). Single worker only:
    # the game registry is in-memory.
    uvicorn.run(
        "settlrl_render.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "1") == "1",
    )
