"""
The LLM Router is the heart of the fallback system.

Responsibilities:
1. At startup, build the list of available providers from env (in fallback order)
2. For each chat call, try providers in order
3. On QuotaExhaustedException:
   a. Mark the provider as exhausted for this session
   b. Log the switch event
   c. Try the next provider with the FULL message history
   d. Update the session's current_provider and exhausted_providers in MongoDB
4. On AllProvidersExhaustedException:
   a. Return a structured error response to the frontend
   b. Do NOT crash the server
5. Process-level exhaustion vs session-level exhaustion:
   - Session-level: a provider hits quota for one session. Other sessions
     can still use it.
   - Process-level: a provider's key is invalid / auth fails permanently.
     Remove it from all sessions.

CRITICAL: When switching providers mid-session, the COMPLETE message history
(from the very first system prompt through all user and assistant messages)
is passed to the new provider. The new provider receives the same context
as if it had been in the conversation from the start.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import structlog

from app.config.settings import Settings
from app.services.llm.base import (
    AllProvidersExhaustedException,
    BaseLLMAdapter,
    ProviderUnavailableException,
    QuotaExhaustedException,
)
from app.services.llm.claude_adapter import ClaudeAdapter
from app.services.llm.gemini_adapter import GeminiAdapter
from app.services.llm.groq_adapter import GroqAdapter
from app.services.llm.openai_adapter import OpenAIAdapter

logger = structlog.get_logger("llm.router")


_ADAPTER_CLASSES: Dict[str, type] = {
    "groq": GroqAdapter,
    "gemini": GeminiAdapter,
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
}


class LLMRouter:
    """Routes LLM calls through available providers with automatic fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._adapters: Dict[str, BaseLLMAdapter] = {}
        self._fallback_order: List[str] = []

        # Build ordered list of provider adapters
        keys = settings.provider_api_keys
        models = settings.provider_models

        for provider_name in settings.fallback_order_list:
            api_key = keys.get(provider_name, "")
            if not api_key or not api_key.strip():
                logger.info(
                    "llm_provider_skipped",
                    provider=provider_name,
                    reason="no_api_key",
                )
                continue

            adapter_cls = _ADAPTER_CLASSES.get(provider_name)
            if adapter_cls is None:
                logger.warning("llm_provider_unknown", provider=provider_name)
                continue

            model = models.get(provider_name, "")
            adapter = adapter_cls(api_key=api_key, model_name=model)
            self._adapters[provider_name] = adapter
            self._fallback_order.append(provider_name)

        logger.info(
            "llm_router_initialized",
            active_providers=self._fallback_order,
            total=len(self._fallback_order),
        )

    @property
    def providers(self) -> Dict[str, BaseLLMAdapter]:
        """Public access to provider adapters dict."""
        return self._adapters

    @property
    def fallback_order(self) -> List[str]:
        """Public access to fallback order."""
        return self._fallback_order

    async def get_available_providers(self) -> List[str]:
        """Returns list of provider names currently available."""
        available: List[str] = []
        for name in self._fallback_order:
            adapter = self._adapters.get(name)
            if adapter and await adapter.is_available():
                available.append(name)
        return available

    def get_all_provider_statuses(self) -> List[Dict[str, str]]:
        """Return status of every known provider (including unconfigured)."""
        keys = self._settings.provider_api_keys
        models = self._settings.provider_models
        statuses: List[Dict[str, str]] = []

        for provider_name in ["groq", "gemini", "claude", "openai"]:
            api_key = keys.get(provider_name, "")
            model = models.get(provider_name, "")
            if not api_key or not api_key.strip():
                status = "no_api_key"
            elif provider_name in self._adapters:
                adapter = self._adapters[provider_name]
                if getattr(adapter, "_process_exhausted", False):
                    status = "process_exhausted"
                else:
                    status = "available"
            else:
                status = "not_configured"
            statuses.append({"name": provider_name, "model": model, "status": status})

        return statuses

    @property
    def active_fallback_order(self) -> List[str]:
        return list(self._fallback_order)

    async def chat(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        session_exhausted_providers: List[str],
        preferred_provider: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> Tuple[str, str, bool, Optional[str]]:
        """
        Send messages to the best available provider with automatic fallback.

        Returns:
            (response_text, provider_used, provider_switched, previous_provider)

        Algorithm:
        1. If preferred_provider is set and available, try it first
        2. Otherwise iterate through provider chain in order
        3. Skip providers in session_exhausted_providers
        4. Skip providers that are process-level unavailable
        5. On QuotaExhaustedException: add to session_exhausted_providers, continue to next
        6. On success: return result with switch metadata
        7. If all fail: raise AllProvidersExhaustedException
        """
        # Build ordered list of providers to try
        providers_to_try: List[str] = []

        if preferred_provider and preferred_provider in self._adapters:
            if preferred_provider not in session_exhausted_providers:
                providers_to_try.append(preferred_provider)

        for name in self._fallback_order:
            if name not in providers_to_try and name not in session_exhausted_providers:
                providers_to_try.append(name)

        if not providers_to_try:
            logger.error(
                "all_providers_exhausted",
                session_id=session_id,
                exhausted=session_exhausted_providers,
            )
            raise AllProvidersExhaustedException(
                "All configured LLM providers are exhausted or unavailable."
            )

        # Separate system messages from conversation messages
        system_parts: List[str] = []
        conversation_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation_messages.append(msg)

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        for idx, provider_name in enumerate(providers_to_try):
            adapter = self._adapters.get(provider_name)
            if adapter is None:
                continue

            if not await adapter.is_available():
                logger.info(
                    "llm_provider_not_available",
                    provider=provider_name,
                    session_id=session_id,
                )
                continue

            try:
                logger.info(
                    "llm_attempting_provider",
                    provider=provider_name,
                    session_id=session_id,
                    attempt=idx + 1,
                )

                response_text = await adapter.chat(
                    messages=conversation_messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                switched = (
                    provider_name != preferred_provider
                    and preferred_provider is not None
                )
                prev = preferred_provider if switched else None

                logger.info(
                    "llm_response_success",
                    provider=provider_name,
                    session_id=session_id,
                    switched=switched,
                )

                return response_text, provider_name, switched, prev

            except QuotaExhaustedException as exc:
                logger.warning(
                    "llm_quota_exhausted_fallback",
                    provider=provider_name,
                    session_id=session_id,
                    error=str(exc),
                )
                session_exhausted_providers.append(provider_name)
                continue

            except ProviderUnavailableException as exc:
                logger.error(
                    "llm_provider_error_fallback",
                    provider=provider_name,
                    session_id=session_id,
                    error=str(exc),
                )
                continue

        # All providers failed
        logger.error(
            "all_providers_failed",
            session_id=session_id,
            tried=providers_to_try,
            exhausted=session_exhausted_providers,
        )
        raise AllProvidersExhaustedException(
            "All configured LLM providers are exhausted or unavailable."
        )
