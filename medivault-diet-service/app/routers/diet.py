"""
All diet plan routes — session CRUD, chat, and plan management.

Base path: /api/v1/diet
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.middleware.auth import CurrentUser, get_current_patient
from app.models.diet_message import DietMessage
from app.schemas.diet import (
    ChatMessageRequest,
    ChatResponse,
    CreateSessionRequest,
    DietPlan,
    MessageItem,
    MessageListResponse,
    SessionListItem,
    SessionListResponse,
    SessionResponse,
)
from app.services import diet_session as session_service
from app.services.diet_chat import (
    create_session_and_generate_plan,
    process_chat_message,
)
from app.services.llm.base import AllProvidersExhaustedException

logger = structlog.get_logger("routers.diet")

router = APIRouter()


def _extract_token(request: Request) -> str:
    """Extract the raw Bearer token from the request headers."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return ""


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    body: CreateSessionRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_patient),
) -> SessionResponse:
    """
    Create a new diet session and generate the initial diet plan.
    Called when the patient taps 'Generate Diet Plan' in the app.

    Auth: JWT required (patient role only)

    Error Responses:
    - 401 — Invalid or expired JWT
    - 403 — Not a patient role
    - 502 — All LLM providers exhausted or unavailable
    - 503 — MediVault main backend unreachable
    """
    jwt_token = _extract_token(request)
    try:
        result = await create_session_and_generate_plan(
            patient_id=current_user.patient_id,
            jwt_token=jwt_token,
            request=body,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except AllProvidersExhaustedException:
        raise  # Handled by global handler → 502


@router.post(
    "/sessions/{session_id}/messages", response_model=ChatResponse
)
async def send_message(
    session_id: str,
    body: ChatMessageRequest,
    current_user: CurrentUser = Depends(get_current_patient),
) -> ChatResponse:
    """
    Send a follow-up message to refine the diet plan.
    Called when the patient sends a message in the chat box.

    Auth: JWT required (patient role, must own this session)

    Error Responses:
    - 401 — Invalid JWT
    - 403 — Session belongs to another patient
    - 404 — Session not found
    - 410 — Session expired
    - 422 — Message too long (> 2000 chars)
    - 429 — Rate limit hit
    - 502 — All LLM providers exhausted
    """
    # Verify session ownership
    session = await session_service.get_session_for_patient(
        session_id, current_user.patient_id
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.patient_id != current_user.patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session belongs to another patient",
        )

    if session.is_expired():
        await session_service.expire_session(session)
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session has expired",
        )

    if session.status != "active":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"Session is not active (status: {session.status})",
        )

    try:
        result = await process_chat_message(
            session_id=session_id,
            patient_id=current_user.patient_id,
            message=body.message,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AllProvidersExhaustedException:
        raise  # Handled by global handler → 502


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_patient),
) -> SessionResponse:
    """
    Fetch the current state of a diet session including the latest diet plan.
    Used to restore session state when the patient reopens the app.

    Auth: JWT required (patient must own session)
    """
    session = await session_service.get_session_for_patient(
        session_id, current_user.patient_id
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    diet_plan = session.current_diet_plan if session.current_diet_plan else None

    return SessionResponse(
        session_id=str(session.id),
        status=session.status,
        diet_plan=diet_plan,
        current_provider=session.current_provider,
        message_count=session.message_count,
        created_at=session.created_at,
    )


@router.get(
    "/sessions/{session_id}/messages", response_model=MessageListResponse
)
async def get_messages(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_patient),
    limit: int = Query(default=50, ge=1, le=100),
    before: Optional[str] = Query(default=None),
) -> MessageListResponse:
    """
    Return the full chat history for a session.
    Used to render the complete conversation in the chat UI.

    Auth: JWT required (patient must own session)

    Query Params:
    - limit: int (default: 50, max: 100)
    - before: ISO datetime string (for pagination)
    """
    session = await session_service.get_session_for_patient(
        session_id, current_user.patient_id
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    query = DietMessage.find(DietMessage.session_id == session_id)

    if before:
        try:
            before_dt = datetime.fromisoformat(
                before.replace("Z", "+00:00")
            )
            query = query.find(DietMessage.created_at < before_dt)
        except (ValueError, TypeError):
            pass

    total = await DietMessage.find(
        DietMessage.session_id == session_id
    ).count()
    messages_docs = (
        await query.sort("+created_at").limit(limit).to_list()
    )

    messages = [
        MessageItem(
            message_id=str(msg.id),
            role=msg.role,
            content=msg.content,
            provider_used=msg.provider_used,
            is_diet_plan=msg.is_diet_plan,
            created_at=msg.created_at,
        )
        for msg in messages_docs
    ]

    return MessageListResponse(
        session_id=session_id,
        messages=messages,
        total=total,
        has_more=len(messages_docs) < total,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    current_user: CurrentUser = Depends(get_current_patient),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=10, ge=1, le=50),
) -> SessionListResponse:
    """
    List all diet sessions for the authenticated patient.
    Used to show session history and allow resuming a previous session.

    Auth: JWT required

    Query Params:
    - status: filter by "active" | "completed" | "expired" (optional)
    - limit: int (default 10)
    """
    sessions, total = await session_service.list_sessions(
        patient_id=current_user.patient_id,
        status=status_filter,
        limit=limit,
    )

    items = [
        SessionListItem(
            session_id=str(s.id),
            status=s.status,
            current_provider=s.current_provider,
            message_count=s.message_count,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]

    return SessionListResponse(sessions=items, total=total)


@router.patch("/sessions/{session_id}/complete")
async def complete_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_patient),
) -> dict:
    """
    Mark a session as completed when the patient is satisfied with their
    diet plan. Called when they tap "Save & Done".

    Auth: JWT required (patient must own session)
    """
    session = await session_service.get_session_for_patient(
        session_id, current_user.patient_id
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    await session_service.complete_session(session)
    return {"session_id": session_id, "status": "completed"}


@router.delete(
    "/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_patient),
):
    """
    Delete a session and all its messages. Called when patient wants
    to start fresh.

    Auth: JWT required (patient must own session)
    """
    session = await session_service.get_session_for_patient(
        session_id, current_user.patient_id
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    await session_service.delete_session(session_id)
