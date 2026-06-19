import os

import uvicorn

from settlrl_app.config import Settings


def main() -> None:
    # The server drives one game at a time; default JAX to CPU so importing it
    # doesn't preallocate most of the GPU's memory. Exported (not just set for
    # this process) so uvicorn's reloader subprocess inherits it; an explicit
    # JAX_PLATFORMS in the environment still wins.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Defaults suit development; deployments override via env (RELOAD=0 in
    # production — the reloader is a dev file-watcher). Single worker only:
    # the game registry is in-memory.
    settings = Settings()
    uvicorn.run(
        "settlrl_app.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
