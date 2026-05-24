import uvicorn


def main() -> None:
    uvicorn.run("catan_render.server:app", host="0.0.0.0", port=8000, reload=True)
