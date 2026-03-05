"""
DietSession — one session per patient "Generate Diet Plan" click.

Stores:
- The patient context snapshot at creation time (never re-fetched mid-session)
- The current diet plan (latest LLM-generated version)
- The active LLM provider name (changes on fallback)
- Session status and expiry
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from beanie import Document, Indexed
from pydantic import Field

from app.config.settings import get_settings


class DietSession(Document):
    patient_id: Indexed(str)  # type: ignore[valid-type]
    patient_jwt: str = ""
    patient_context: Dict[str, Any] = {}
    current_diet_plan: Optional[Dict[str, Any]] = None
    current_provider: str = ""
    exhausted_providers: List[str] = Field(default_factory=list)
    message_count: int = 0
    vector_chunks_count: int = 0
    vector_context_used: bool = False
    cuisine_preference: str = "indian"
    regional_preference: Optional[str] = None
    status: Literal["active", "completed", "expired", "error"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(hours=get_settings().session_inactivity_timeout_hours)
    )

    class Settings:
        name = "diet_sessions"
        indexes = [
            "patient_id",
            "status",
            "expires_at",
        ]

    def is_expired(self) -> bool:
        exp = self.expires_at
        # Ensure both sides are offset-aware for comparison
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp

    def touch(self) -> None:
        """Update timestamps and slide the expiry window."""
        now = datetime.now(timezone.utc)
        settings = get_settings()
        self.updated_at = now
        self.expires_at = now + timedelta(hours=settings.session_inactivity_timeout_hours)
