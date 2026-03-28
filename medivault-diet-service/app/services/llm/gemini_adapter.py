"""
Google Gemini provider implementation using google-generativeai SDK.

IMPORTANT: Gemini has a different message format. Converts OpenAI-style
messages to Gemini format:
  - "system" role messages -> system_instruction parameter
  - "user" -> "user"
  - "assistant" -> "model"
Detects quota errors: google.api_core.exceptions.ResourceExhausted and 429 status codes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import structlog

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
except ImportError:
    genai = None  # type: ignore[assignment]
    ResourceExhausted = Exception  # type: ignore[assignment,misc]
    GoogleAPIError = Exception  # type: ignore[assignment,misc]

from app.services.llm.base import (
    BaseLLMAdapter,
    ProviderUnavailableException,
    QuotaExhaustedException,
)

logger = structlog.get_logger("llm.gemini")


class GeminiAdapter(BaseLLMAdapter):
    provider_name: str = "gemini"

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._process_exhausted = False
        self._configured = False
        if api_key and api_key.strip() and genai is not None:
            genai.configure(api_key=api_key)
            self._configured = True

    async def is_available(self) -> bool:
        if not self._api_key or not self._api_key.strip():
            return False
        if self._process_exhausted:
            return False
        if genai is None:
            return False
        return True

    def _convert_messages(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str]
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini format.
        System messages are prepended to the system instruction.
        """
        system_parts: List[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        gemini_history: list[dict] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})
            else:
                gemini_history.append({"role": "user", "parts": [content]})

        system_instruction = "\n\n".join(system_parts) if system_parts else ""
        return system_instruction, gemini_history

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        if genai is None:
            raise ProviderUnavailableException(
                provider="gemini", message="google-generativeai not installed"
            )

        system_instruction, gemini_history = self._convert_messages(messages, system_prompt)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction if system_instruction else None,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            # Gemini expects alternating user/model. The last message MUST be user.
            # Ensure history alternates properly and ends with user.
            if gemini_history and gemini_history[-1]["role"] != "user":
                # If last message isn't user, append a continuation prompt
                gemini_history.append({"role": "user", "parts": ["Please continue with the diet plan."]})

            if len(gemini_history) > 1:
                # Start chat with history minus the last user message
                chat_history = gemini_history[:-1]
                last_message = gemini_history[-1]

                chat = model.start_chat(history=chat_history)
                response = await chat.send_message_async(last_message["parts"][0])
            elif gemini_history:
                response = await model.generate_content_async(
                    gemini_history[0]["parts"][0]
                )
            else:
                raise ProviderUnavailableException(
                    provider="gemini", message="No messages provided"
                )

            content = response.text or ""
            logger.info("gemini_response", model=self.model_name)
            return content

        except ResourceExhausted as exc:
            logger.warning("gemini_quota_exhausted", error=str(exc))
            raise QuotaExhaustedException(provider="gemini", message=str(exc)) from exc

        except GoogleAPIError as exc:
            error_str = str(exc)
            if "429" in error_str or "quota" in error_str.lower():
                raise QuotaExhaustedException(
                    provider="gemini", message=error_str
                ) from exc
            if "401" in error_str or "403" in error_str:
                self._process_exhausted = True
                logger.error("gemini_auth_failure", error=error_str)
            raise ProviderUnavailableException(
                provider="gemini", message=error_str
            ) from exc

        except Exception as exc:
            # Avoid wrapping our own exceptions
            if isinstance(exc, (QuotaExhaustedException, ProviderUnavailableException)):
                raise
            logger.error("gemini_unexpected_error", error=str(exc))
            raise ProviderUnavailableException(
                provider="gemini", message=str(exc)
            ) from exc
