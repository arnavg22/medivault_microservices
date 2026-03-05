"""
Process-level provider exhaustion state tracker.

Keeps an in-memory record of which LLM providers have been marked as
exhausted (quota used up) so the LLMRouter can skip them on subsequent
requests without re-trying.

This is process-local (not shared across workers). Each Gunicorn/Uvicorn
worker maintains its own state. A provider auto-recovers after a
configurable cooldown period.
"""

from __future__ import annotations

import time
from typing import Dict

import structlog

logger = structlog.get_logger("utils.provider_state")

# Default cooldown: 5 minutes before retrying an exhausted provider
DEFAULT_COOLDOWN_SECONDS = 300

# { provider_name: timestamp_when_exhausted }
_exhausted_providers: Dict[str, float] = {}
_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS


def set_cooldown(seconds: float) -> None:
    """Override the global cooldown period (useful in tests)."""
    global _cooldown_seconds
    _cooldown_seconds = seconds


def mark_exhausted(provider: str) -> None:
    """
    Mark a provider as exhausted. It will be skipped until the
    cooldown period elapses.
    """
    _exhausted_providers[provider] = time.time()
    logger.warning("provider_marked_exhausted", provider=provider)


def is_exhausted(provider: str) -> bool:
    """
    Check whether a provider is currently in the exhausted state.
    Automatically clears the flag if the cooldown has elapsed.
    """
    if provider not in _exhausted_providers:
        return False

    elapsed = time.time() - _exhausted_providers[provider]
    if elapsed >= _cooldown_seconds:
        _exhausted_providers.pop(provider, None)
        logger.info(
            "provider_cooldown_elapsed",
            provider=provider,
            elapsed_seconds=round(elapsed, 1),
        )
        return False

    return True


def get_all_exhausted() -> Dict[str, float]:
    """Return a snapshot of all currently exhausted providers."""
    now = time.time()
    return {
        name: round(now - ts, 1)
        for name, ts in _exhausted_providers.items()
        if (now - ts) < _cooldown_seconds
    }


def clear_all() -> None:
    """Reset all exhaustion state (useful in tests)."""
    _exhausted_providers.clear()
