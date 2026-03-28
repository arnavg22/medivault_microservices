"""
Anthropic Claude provider implementation using the official anthropic SDK (async).

IMPORTANT: Claude separates the system prompt from messages. Extract
messages with role "system" and pass them as the `system` parameter.
All other messages go in the `messages` array.
Detects quota errors: anthropic.RateLimitError and HTTP 429.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import structlog

try:
    from anthropic import (
        AsyncAnthropic,
        RateLimitError as AnthropicRateLimitError,
        APIStatusError as AnthropicAPIStatusError,
    )
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]
    AnthropicRateLimitError = Exception  # type: ignore[assignment,misc]
    AnthropicAPIStatusError = Exception  # type: ignore[assignment,misc]

from app.services.llm.base import (
    BaseLLMAdapter,
    ProviderUnavailableException,
    QuotaExhaustedException,
)

logger = structlog.get_logger("llm.claude")


class ClaudeAdapter(BaseLLMAdapter):
    provider_name: str = "claude"

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client: "AsyncAnthropic | None" = None
        self._process_exhausted = False

    def _get_client(self) -> "AsyncAnthropic":
        if self._client is None:
            if AsyncAnthropic is None:
                raise ProviderUnavailableException(
                    provider="claude", message="anthropic SDK not installed"
                )
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def is_available(self) -> bool:
        if not self._api_key or not self._api_key.strip():
            return False
        if self._process_exhausted:
            return False
        if AsyncAnthropic is None:
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

        # Claude separates system from messages
        system_parts: List[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        claude_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

        system_text = "\n\n".join(system_parts) if system_parts else ""

        # Claude requires messages to start with a user message.
        # If first message is assistant, prepend the system context as a user turn
        # so Claude has proper context rather than a confusing "Please continue."
        if claude_messages and claude_messages[0]["role"] != "user":
            prefix = system_text[:200] + "..." if len(system_text) > 200 else system_text
            claude_messages.insert(0, {"role": "user", "content": prefix or "Generate the diet plan based on my preferences."})

        try:
            response = await client.messages.create(
                model=self.model_name,
                system=system_text,
                messages=claude_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.content[0].text if response.content else ""
            logger.info(
                "claude_response",
                model=self.model_name,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            return content

        except AnthropicRateLimitError as exc:
            logger.warning("claude_quota_exhausted", error=str(exc))
            raise QuotaExhaustedException(provider="claude", message=str(exc)) from exc

        except AnthropicAPIStatusError as exc:
            error_str = str(exc)
            if hasattr(exc, "status_code") and exc.status_code == 429:
                raise QuotaExhaustedException(
                    provider="claude", message=error_str
                ) from exc
            if hasattr(exc, "status_code") and exc.status_code in (401, 403):
                self._process_exhausted = True
                logger.error("claude_auth_failure", error=error_str)
            raise ProviderUnavailableException(
                provider="claude", message=error_str
            ) from exc

        except Exception as exc:
            if isinstance(exc, (QuotaExhaustedException, ProviderUnavailableException)):
                raise
            logger.error("claude_unexpected_error", error=str(exc))
            raise ProviderUnavailableException(
                provider="claude", message=str(exc)
            ) from exc
