"""
MediVault Diet Service — FastAPI Application Factory.

This is the main entry point that wires everything together:
  - Database lifecycle (init / close)
  - LLM router initialisation
  - CORS middleware
  - Rate limiting
  - Exception handlers
  - Router mounting
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config.database import close_db, init_db
from app.config.settings import get_settings
from app.middleware.error_handler import register_exception_handlers
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.routers import diet, health
from app.services.diet_chat import set_llm_router
from app.services.llm.router import LLMRouter
from app.utils.logger import setup_logging

logger = structlog.get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.

    On startup:
      1. Configure structured logging
      2. Connect to MongoDB and initialise Beanie ODM
      3. Build the LLM provider chain

    On shutdown:
      1. Close the MongoDB connection pool
    """
    settings = get_settings()

    # 1. Logging
    setup_logging(
        log_level=settings.log_level,
        json_output=(settings.node_env == "production"),
    )
    logger.info(
        "starting_diet_service",
        environment=settings.node_env,
        port=settings.port,
    )

    # 2. Database
    await init_db()
    logger.info("database_connected")

    # 3. LLM Router
    llm_router = LLMRouter(settings)
    set_llm_router(llm_router)
    provider_names = list(llm_router.providers.keys())
    logger.info(
        "llm_router_initialised",
        providers=provider_names,
        fallback_order=list(llm_router.fallback_order),
    )

    yield

    # Shutdown
    await close_db()
    logger.info("diet_service_shutdown_complete")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MediVault Diet Service",
        description=(
            "AI-powered personalised diet plan generator for MediVault "
            "Hospital Management System. Generates clinically-aware "
            "7-day diet plans using patient medical records."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/v1/diet/docs" if settings.node_env != "production" else None,
        redoc_url="/api/v1/diet/redoc" if settings.node_env != "production" else None,
        openapi_url="/api/v1/diet/openapi.json" if settings.node_env != "production" else None,
    )

    # ── CORS ──────────────────────────────────────────────────────
    allowed_origins = settings.cors_origins_list

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID ────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)

    # ── Security Headers ──────────────────────────────────────────
    app.add_middleware(SecurityHeadersMiddleware)

    # ── Rate Limiting ─────────────────────────────────────────────
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Exception Handlers ────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────
    app.include_router(
        health.router,
        prefix="/api/v1/diet",
        tags=["Health"],
    )
    app.include_router(
        diet.router,
        prefix="/api/v1/diet",
        tags=["Diet"],
    )

    return app


# Module-level app instance used by Uvicorn / Gunicorn
app = create_app()
