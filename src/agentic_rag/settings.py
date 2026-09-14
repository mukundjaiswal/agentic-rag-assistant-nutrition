"""Typed, validated, environment-driven configuration.

Validation happens once at load. A threshold outside ``[0, 1]`` or a
non-positive iteration cap fails immediately with a readable error rather than
producing a graph that silently never terminates.

Secrets are held as :class:`~pydantic.SecretStr` so that logging or repr-ing a
settings object cannot leak an API key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- LLM provider -------------------------------------------------------
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Document parsing ---------------------------------------------------
    llama_cloud_api_key: SecretStr | None = None

    # --- Vector store -------------------------------------------------------
    chroma_dir: Path = Path("./.chroma")
    collection_name: str = "corpus"
    retrieval_k: int = Field(default=5, ge=1, le=50)

    # --- Graph control ------------------------------------------------------
    groundedness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    precision_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_iterations: int = Field(default=3, ge=1, le=10)

    # --- Safety -------------------------------------------------------------
    enable_safety_filter: bool = False
    safety_model: str | None = None

    # --- Observability ------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    @field_validator("collection_name")
    @classmethod
    def _collection_name_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            message = "collection_name must not be blank"
            raise ValueError(message)
        return value

    def require_openai_api_key(self) -> str:
        """Return the API key, or fail with an actionable message.

        Deferred to call time rather than enforced at load, so that unit tests
        and ``--help`` do not require credentials.
        """
        if self.openai_api_key is None:
            message = (
                "AGENTIC_RAG_OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill it in."
            )
            raise ConfigurationError(message)
        return self.openai_api_key.get_secret_value()

    def require_llama_cloud_api_key(self) -> str:
        """Return the document-parsing API key, or fail with a clear message."""
        if self.llama_cloud_api_key is None:
            message = (
                "AGENTIC_RAG_LLAMA_CLOUD_API_KEY is not set; document parsing "
                "cannot run. Copy .env.example to .env and fill it in."
            )
            raise ConfigurationError(message)
        return self.llama_cloud_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so that configuration is read once. Call ``get_settings.cache_clear()``
    in tests that need to vary the environment.
    """
    return Settings()
