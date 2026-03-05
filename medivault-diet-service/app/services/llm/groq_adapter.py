"""
Groq provider implementation using the official groq Python SDK (async).

Uses AsyncGroq client. Messages format is OpenAI-compatible (Groq uses the same format).
Detects quota errors on HTTP 429 / error code rate_limit_exceeded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import structlog
from groq import AsyncGroq, APIStatusError, RateLimitError

from app.services.llm.base import (
    BaseLLMAdapter,
    ProviderUnavailableException,
    QuotaExhaustedException,
)

logger = structlog.get_logger("llm.groq")


class GroqAdapter(BaseLLMAdapter):
    provider_name: str = "groq"

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client: AsyncGroq | None = None
        self._process_exhausted = False

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    async def is_available(self) -> bool:
        if not self._api_key or not self._api_key.strip():
            return False
        if self._process_exhausted:
            return False
        return True

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        client = self._get_client()

        formatted: List[Dict[str, str]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend(messages)

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=formatted,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.info(
                "groq_response",
                model=self.model_name,
                tokens_used=getattr(response.usage, "total_tokens", None),
            )
            return content

        except RateLimitError as exc:
            logger.warning("groq_quota_exhausted", error=str(exc))
            raise QuotaExhaustedException(provider="groq", message=str(exc)) from exc

        except APIStatusError as exc:
            if exc.status_code == 401:
                self._process_exhausted = True
                logger.error("groq_auth_failure", error=str(exc))
            raise ProviderUnavailableException(provider="groq", message=str(exc)) from exc

        except Exception as exc:
            logger.error("groq_unexpected_error", error=str(exc))
            raise ProviderUnavailableException(provider="groq", message=str(exc)) from exc
