"""Bot-catalog routes: the public ``GET /api/bots`` plus the admin-only
``/api/admin/bot-providers`` CRUD over registered remote bot services.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from settlrl_app.api.deps import Deps
from settlrl_app.bots.providers import RemoteBotError


class _ProviderRequest(BaseModel):
    """A remote bot service to register: its base URL (the bot self-identifies)."""

    base_url: str


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    bots, auth = deps.bots, deps.auth

    @router.get("/api/bots")
    def get_bots() -> dict[str, dict[str, object]]:
        """Bot kinds available for seats — one per registered bot service — each
        with its title, description, and the player counts it supports."""
        return bots.catalog()

    @router.get("/api/admin/bot-providers")
    def list_bot_providers(
        _: Annotated[object, Depends(auth.admin_user)],
    ) -> list[dict[str, object]]:
        """Registered remote bot providers (admin only)."""
        return bots.providers()

    @router.post("/api/admin/bot-providers", status_code=201)
    async def register_bot_provider(
        req: _ProviderRequest, _: Annotated[object, Depends(auth.admin_user)]
    ) -> dict[str, object]:
        """Register (or replace) a remote bot service by base URL; the bot it
        serves joins the catalog under its own name. ``400`` if it is unreachable
        (admin only)."""
        try:
            provider = await bots.register(req.base_url)
        except RemoteBotError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"name": provider.name, "base_url": provider.base_url}

    @router.delete("/api/admin/bot-providers/{name}", status_code=204)
    def remove_bot_provider(
        name: str, _: Annotated[object, Depends(auth.admin_user)]
    ) -> None:
        """Unregister a remote bot provider (admin only); ``404`` if unknown."""
        if not bots.unregister(name):
            raise HTTPException(status_code=404, detail="no such provider")

    return router
