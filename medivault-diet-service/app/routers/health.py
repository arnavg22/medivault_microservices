"""
Health-check and provider status routes.

These endpoints live at:
  GET /api/v1/diet/health       (no auth — for uptime monitors)
  GET /api/v1/diet/providers    (auth required — for debug UI)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from app.middleware.auth import CurrentUser, get_current_patient
from app.schemas.diet import HealthResponse, ProvidersResponse
from app.services.diet_chat import get_llm_router

logger = structlog.get_logger("routers.health")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Lightweight health probe. Returns 200 if the process is alive.
    Used by Render.com / PM2 health checks.
    No authentication required.
    """
    return HealthResponse(status="ok", service="medivault-diet-service")


@router.get("/providers", response_model=ProvidersResponse)
async def provider_status(
    current_user: CurrentUser = Depends(get_current_patient),
) -> ProvidersResponse:
    """
    Show which LLM providers are currently available and their
    priority order. Useful for debugging and the admin panel.

    Auth: JWT required
    """
    llm_router = get_llm_router()
    if llm_router is None:
        return ProvidersResponse(
            providers=[],
            active_provider=None,
            fallback_chain=[],
        )

    provider_list = []
    for name, adapter in llm_router.providers.items():
        provider_list.append(
            {
                "name": name,
                "available": await adapter.is_available(),
                "model": getattr(adapter, "model_name", "unknown"),
            }
        )

    active = None
    for name in llm_router.fallback_order:
        adapter = llm_router.providers.get(name)
        if adapter and await adapter.is_available():
            active = name
            break

    return ProvidersResponse(
        providers=provider_list,
        active_provider=active,
        fallback_chain=list(llm_router.fallback_order),
    )
