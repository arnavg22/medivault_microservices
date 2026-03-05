"""
Session CRUD and lifecycle management for diet sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import structlog
from beanie import PydanticObjectId

from app.models.diet_message import DietMessage
from app.models.diet_session import DietSession

logger = structlog.get_logger("diet_session")


async def create_session(
    patient_id: str,
    patient_jwt: str,
    patient_context: dict,
    provider: str,
    cuisine_preference: str = "indian",
    regional_preference: Optional[str] = None,
) -> DietSession:
    """Create a new diet session."""
    session = DietSession(
        patient_id=patient_id,
        patient_jwt=patient_jwt,
        patient_context=patient_context,
        current_provider=provider,
        cuisine_preference=cuisine_preference,
        regional_preference=regional_preference,
    )
    await session.insert()
    logger.info(
        "session_created", session_id=str(session.id), patient_id=patient_id
    )
    return session


async def get_session(session_id: str) -> Optional[DietSession]:
    """Fetch a session by ID."""
    try:
        return await DietSession.get(PydanticObjectId(session_id))
    except Exception:
        return None


async def get_session_for_patient(
    session_id: str, patient_id: str
) -> Optional[DietSession]:
    """Fetch a session and verify ownership."""
    session = await get_session(session_id)
    if session and session.patient_id == patient_id:
        return session
    return None


async def list_sessions(
    patient_id: str,
    status: Optional[str] = None,
    limit: int = 10,
) -> tuple[List[DietSession], int]:
    """List sessions for a patient with optional status filter."""
    query: dict = {"patient_id": patient_id}
    if status:
        query["status"] = status

    total = await DietSession.find(query).count()
    sessions = (
        await DietSession.find(query).sort("-created_at").limit(limit).to_list()
    )
    return sessions, total


async def update_session_provider(
    session: DietSession,
    new_provider: str,
    old_provider: str,
) -> None:
    """Update session after a provider switch."""
    session.current_provider = new_provider
    if old_provider and old_provider not in session.exhausted_providers:
        session.exhausted_providers.append(old_provider)
    session.touch()
    await session.save()
    logger.info(
        "session_provider_switched",
        session_id=str(session.id),
        old_provider=old_provider,
        new_provider=new_provider,
    )


async def update_session_plan(
    session: DietSession, plan_dict: Optional[dict], provider: str
) -> None:
    """Update the current diet plan on the session."""
    session.current_diet_plan = plan_dict
    session.current_provider = provider
    session.touch()
    await session.save()


async def increment_message_count(
    session: DietSession, count: int = 1
) -> None:
    """Increment message count and touch session."""
    session.message_count += count
    session.touch()
    await session.save()


async def complete_session(session: DietSession) -> None:
    """Mark session as completed."""
    session.status = "completed"
    session.updated_at = datetime.now(timezone.utc)
    await session.save()
    logger.info("session_completed", session_id=str(session.id))


async def expire_session(session: DietSession) -> None:
    """Mark session as expired."""
    session.status = "expired"
    session.updated_at = datetime.now(timezone.utc)
    await session.save()
    logger.info("session_expired", session_id=str(session.id))


async def delete_session(session_id: str) -> None:
    """Delete a session and all its messages."""
    # Delete all messages
    await DietMessage.find(DietMessage.session_id == session_id).delete()
    # Delete session
    session = await get_session(session_id)
    if session:
        await session.delete()
    logger.info("session_deleted", session_id=session_id)
