"""
JWT authentication middleware.

Extracts and validates JWTs using the SAME secret as the Node.js
MediVault backend, so a token issued there works here.
"""

from __future__ import annotations

from typing import Optional

import structlog
from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config.settings import get_settings

logger = structlog.get_logger("middleware.auth")


class CurrentUser(BaseModel):
    """Decoded payload carried in every authenticated request."""

    user_id: str
    patient_id: str
    role: str
    email: Optional[str] = None


async def get_current_patient(request: Request) -> CurrentUser:
    """
    FastAPI dependency — extracts the JWT from the Authorization header,
    decodes it, verifies the role is 'patient', and returns a CurrentUser.

    Raises:
    - 401 if the token is missing, malformed, or expired
    - 403 if the role is not 'patient'
    """
    settings = get_settings()

    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]

    try:
        payload = jwt.decode(
            token,
            settings.jwt_access_secret,
            algorithms=[settings.jwt_algorithm],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        logger.warning("jwt_decode_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # Extract fields — adapt keys to match your Node backend's JWT payload
    user_id = payload.get("id") or payload.get("userId") or payload.get("sub", "")
    patient_id = payload.get("patientId") or payload.get("patient_id") or user_id
    role = (payload.get("role") or "").lower()
    email = payload.get("email")

    if role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{role}' is not allowed. Patient role required.",
        )

    return CurrentUser(
        user_id=user_id,
        patient_id=patient_id,
        role=role,
        email=email,
    )
