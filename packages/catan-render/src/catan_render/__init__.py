import os

import uvicorn


def main() -> None:
    # The server drives one game at a time; default JAX to CPU so importing it
    # doesn't preallocate most of the GPU's memory. Exported (not just set for
    # this process) so uvicorn's reloader subprocess inherits it; an explicit
    # JAX_PLATFORMS in the environment still wins.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    uvicorn.run("catan_render.server:app", host="0.0.0.0", port=8000, reload=True)
