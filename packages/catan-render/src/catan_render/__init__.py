import os
import sys

import uvicorn


def _create_key_warning() -> str | None:
    """The operator startup warning to show, or None when nothing's wrong.

    Open game creation only matters for a reachable deployment, so this stays
    quiet in dev/hotseat (the default ``RELOAD=1``, where keyless creation is
    the point) and speaks up only on a production run (``RELOAD=0``) with no
    ``CATAN_RENDER_CREATE_KEY`` set (empty counts as unset).
    """
    if os.environ.get("RELOAD", "1") == "1":
        return None
    if os.environ.get("CATAN_RENDER_CREATE_KEY"):
        return None
    return (
        "CATAN_RENDER_CREATE_KEY is not set: game creation is open to anyone who "
        "can reach this server. Set it (CATAN_CREATE_KEY in infra/.env) before "
        "exposing the site publicly."
    )


def main() -> None:
    # The server drives one game at a time; default JAX to CPU so importing it
    # doesn't preallocate most of the GPU's memory. Exported (not just set for
    # this process) so uvicorn's reloader subprocess inherits it; an explicit
    # JAX_PLATFORMS in the environment still wins.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    warning = _create_key_warning()
    if warning is not None:
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)
    # Defaults suit development; deployments override via env (RELOAD=0 in
    # production — the reloader is a dev file-watcher). Single worker only:
    # the game registry is in-memory.
    uvicorn.run(
        "catan_render.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "1") == "1",
    )
