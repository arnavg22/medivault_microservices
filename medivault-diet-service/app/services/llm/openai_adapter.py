"""
OpenAI provider implementation using the official openai SDK (async).

Uses AsyncOpenAI client. Messages format is OpenAI-compatible (no conversion needed).
Detects quota errors: openai.RateLimitError.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import structlog

try:
    from openai import (
        AsyncOpenAI,
        RateLimitError as OpenAIRateLimitError,
        APIStatusError as OpenAIAPIStatusError,
    )
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    OpenAIRateLimitError = Exception  # type: ignore[assignment,misc]
    OpenAIAPIStatusError = Exception  # type: ignore[assignment,misc]

from app.services.llm.base import (
    BaseLLMAdapter,
    ProviderUnavailableException,
    QuotaExhaustedException,
)

logger = structlog.get_logger("llm.openai")


class OpenAIAdapter(BaseLLMAdapter):
    provider_name: str = "openai"

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client: "AsyncOpenAI | None" = None
        self._process_exhausted = False

    def _get_client(self) -> "AsyncOpenAI":
        if self._client is None:
            if AsyncOpenAI is None:
                raise ProviderUnavailableException(
                    provider="openai", message="openai SDK not installed"
                )
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def is_available(self) -> bool:
        if not self._api_key or not self._api_key.strip():
            return False
        if self._process_exhausted:
            return False
        if AsyncOpenAI is None:
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
                "openai_response",
                model=self.model_name,
                tokens_used=getattr(response.usage, "total_tokens", None),
            )
            return content

        except OpenAIRateLimitError as exc:
            logger.warning("openai_quota_exhausted", error=str(exc))
            raise QuotaExhaustedException(provider="openai", message=str(exc)) from exc

        except OpenAIAPIStatusError as exc:
            error_str = str(exc)
            if hasattr(exc, "status_code") and exc.status_code == 429:
                raise QuotaExhaustedException(
                    provider="openai", message=error_str
                ) from exc
            if hasattr(exc, "status_code") and exc.status_code in (401, 403):
                self._process_exhausted = True
                logger.error("openai_auth_failure", error=error_str)
            raise ProviderUnavailableException(
                provider="openai", message=error_str
            ) from exc

        except Exception as exc:
            if isinstance(exc, (QuotaExhaustedException, ProviderUnavailableException)):
                raise
            logger.error("openai_unexpected_error", error=str(exc))
            raise ProviderUnavailableException(
                provider="openai", message=str(exc)
            ) from exc
