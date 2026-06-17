"""Bot-catalog routes: the public ``GET /api/bots`` plus the admin-only
``/api/admin/bot-providers`` CRUD over registered remote bot services.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from settlrl_render.api.deps import Deps
from settlrl_render.bots.providers import RemoteBotError


class _ProviderRequest(BaseModel):
    """A remote bot service to register: a short name and its base URL."""

    name: str
    base_url: str


def build(deps: Deps) -> APIRouter:
    router = APIRouter()
    bots, auth = deps.bots, deps.auth

    @router.get("/api/bots")
    def get_bots() -> dict[str, dict[str, object]]:
        """Bot kinds available for seats — built-in (settlrl-agents) names plus
        any registered remote providers' — each with the player counts it
        supports and its configurable build parameters."""
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
        """Register (or replace) a remote bot service by name + base URL; its
        bot kinds join the catalog. ``400`` if it is unreachable or a kind
        clashes with an existing one (admin only)."""
        try:
            provider = await bots.register(req.name, req.base_url)
        except RemoteBotError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "name": provider.name,
            "base_url": provider.base_url,
            "kinds": sorted(provider.kinds),
        }

    @router.delete("/api/admin/bot-providers/{name}", status_code=204)
    def remove_bot_provider(
        name: str, _: Annotated[object, Depends(auth.admin_user)]
    ) -> None:
        """Unregister a remote bot provider (admin only); ``404`` if unknown."""
        if not bots.unregister(name):
            raise HTTPException(status_code=404, detail="no such provider")

    return router
