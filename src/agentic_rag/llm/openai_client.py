"""OpenAI-backed implementation of :class:`~agentic_rag.interfaces.ChatModel`.

The vendor SDK is imported inside the constructor so that importing this module
costs nothing and requires no credentials. Tests import the package freely; only
an actual instantiation needs a key.
"""

from __future__ import annotations

from typing import Any

from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings

logger = get_logger(__name__)


class OpenAIChatModel:
    """Adapter around a chat-completions endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the client, validating credentials eagerly."""
        from langchain_openai import ChatOpenAI

        self._settings = settings or get_settings()
        self._default_temperature = self._settings.llm_temperature
        self._client: Any = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=self._default_temperature,
            api_key=self._settings.require_openai_api_key(),
        )

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        """Return the model's completion for ``prompt``."""
        client = self._client
        if temperature is not None and temperature != self._default_temperature:
            client = self._client.bind(temperature=temperature)
        response = client.invoke(prompt)
        text = getattr(response, "content", response)
        logger.debug(
            "llm.complete", extra={"chars_in": len(prompt), "chars_out": len(str(text))}
        )
        return str(text).strip()
