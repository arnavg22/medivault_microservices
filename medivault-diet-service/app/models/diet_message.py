"""
DietMessage — each individual message turn in a diet session.

One document per message turn. The full message history is reconstructed
by querying all messages for a session_id ordered by created_at.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from beanie import Document, Indexed
from pydantic import Field


class DietMessage(Document):
    session_id: Indexed(str)  # type: ignore[valid-type]
    role: Literal["user", "assistant", "system"]
    content: str
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    is_diet_plan: bool = False
    token_count: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "diet_messages"
        indexes = [
            "session_id",
            "created_at",
            "is_diet_plan",
        ]
