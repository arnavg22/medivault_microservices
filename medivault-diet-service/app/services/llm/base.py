"""
Abstract base class and exceptions for all LLM provider adapters.

Every LLM provider adapter must inherit from BaseLLMAdapter.
The router only calls these methods — it does not know which provider it is using.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class QuotaExhaustedException(Exception):
    """Raised when a provider returns a rate limit or quota error."""

    def __init__(self, provider: str, message: str = ""):
        self.provider = provider
        super().__init__(f"Quota exhausted for {provider}: {message}")


class ProviderUnavailableException(Exception):
    """Raised for any other provider error (auth failure, server error, etc.)."""

    def __init__(self, provider: str, message: str = ""):
        self.provider = provider
        super().__init__(f"Provider {provider} unavailable: {message}")


class AllProvidersExhaustedException(Exception):
    """Raised when ALL configured providers have failed or hit quota."""

    pass


class BaseLLMAdapter(ABC):
    """
    Every LLM provider adapter must inherit from this class.
    The router only calls these methods — it does not know which provider it is using.
    """

    provider_name: str
    model_name: str

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if this provider can accept requests right now.
        Returns False if:
        - API key is not configured (blank in .env)
        - Provider has been marked as exhausted in this process
        - A quick ping/validation fails
        Never throws — always returns bool.
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        """
        Send a full conversation history to the LLM and get a response.

        IMPORTANT: All providers must accept OpenAI-style message format
        as input. Each adapter internally converts to its own API format.

        The messages list includes the FULL history from the beginning of
        the session — this is what enables context preservation on fallback.

        Raises:
            QuotaExhaustedException: when rate limit / quota is hit
            ProviderUnavailableException: for any other provider error
        """
        ...
