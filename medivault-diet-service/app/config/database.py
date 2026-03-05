"""
Motor + Beanie initialisation and connection lifecycle.
"""

from __future__ import annotations

import structlog
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.config.settings import get_settings

logger = structlog.get_logger("database")

_client: AsyncIOMotorClient | None = None


async def init_db() -> None:
    """Initialise Motor client and Beanie ODM with document models."""
    global _client
    settings = get_settings()

    _client = AsyncIOMotorClient(settings.mongodb_uri)
    db = _client[settings.mongodb_db_name]

    # Import document models here to avoid circular imports
    from app.models.diet_message import DietMessage
    from app.models.diet_session import DietSession

    await init_beanie(database=db, document_models=[DietSession, DietMessage])
    logger.info("mongodb_connected", database=settings.mongodb_db_name)


async def close_db() -> None:
    """Close the Motor client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("mongodb_disconnected")


def get_client() -> AsyncIOMotorClient | None:
    return _client


def get_db():
    """Return the Motor database instance for the diet service."""
    if _client is None:
        return None
    settings = get_settings()
    return _client[settings.mongodb_db_name]
