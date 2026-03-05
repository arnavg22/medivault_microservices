"""
Pydantic BaseSettings class. Reads from .env file.
Validates at startup — if no LLM API keys are configured,
raise a startup error with a clear message.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Server ────────────────────────────────────────────────────────────
    node_env: str = "development"
    port: int = 5001
    log_level: str = "info"

    # ── MongoDB ───────────────────────────────────────────────────────────
    mongodb_uri: str
    mongodb_db_name: str = "medivault_diet"

    # ── JWT ───────────────────────────────────────────────────────────────
    jwt_access_secret: str
    jwt_algorithm: str = "HS256"

    # ── MediVault Backend ─────────────────────────────────────────────────
    medivault_api_base_url: str = ""  # Set via MEDIVAULT_API_BASE_URL in .env

    # ── LLM Provider API Keys ────────────────────────────────────────────
    groq_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # ── LLM Model Names ──────────────────────────────────────────────────
    groq_model: str = "llama3-70b-8192"
    gemini_model: str = "gemini-1.5-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    openai_model: str = "gpt-4o-mini"

    # ── LLM Fallback Order ───────────────────────────────────────────────
    llm_fallback_order: str = "groq,gemini,claude,openai"

    # ── Vector Store (Atlas Vector Search) ────────────────────────────
    mongodb_collection_name: str = "medical_vectors"
    vector_index_name: str = "vector_index"
    vector_search_num_results: int = 8
    # How many vector chunks to retrieve per patient query.
    # 8 gives enough clinical context without overflowing the LLM context window.

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # Must produce embeddings matching the RAG bot's vector index dimensions (768).
    # If OPENAI_API_KEY is set, uses text-embedding-3-small (dim=768 to match index).
    # If GEMINI_API_KEY is set, uses text-embedding-004 (768 dim natively).
    # Otherwise falls back to local sentence-transformers with this model.

    # ── Session Config ───────────────────────────────────────────────────
    session_max_messages: int = 100
    session_inactivity_timeout_hours: int = 24
    max_context_tokens: int = 6000

    # ── Rate Limiting ────────────────────────────────────────────────────
    rate_limit_requests_per_minute: int = 30
    rate_limit_chat_per_minute: int = 15

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_allowed_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:8081"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Computed helpers ─────────────────────────────────────────────────

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def fallback_order_list(self) -> List[str]:
        return [p.strip().lower() for p in self.llm_fallback_order.split(",") if p.strip()]

    @property
    def provider_api_keys(self) -> dict[str, str]:
        return {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "claude": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }

    @property
    def provider_models(self) -> dict[str, str]:
        return {
            "groq": self.groq_model,
            "gemini": self.gemini_model,
            "claude": self.anthropic_model,
            "openai": self.openai_model,
        }

    @property
    def active_providers(self) -> List[str]:
        """Return providers from fallback order that have non-empty API keys."""
        keys = self.provider_api_keys
        return [p for p in self.fallback_order_list if keys.get(p, "").strip()]

    @model_validator(mode="after")
    def validate_at_least_one_provider(self) -> "Settings":
        if not self.active_providers:
            raise ValueError(
                "No LLM providers are configured. Set at least one of: "
                "GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY"
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()  # type: ignore[call-arg]
