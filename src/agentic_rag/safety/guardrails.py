"""Input safety filter. Disabled by default, and deliberately so.

The original implementation called a hosted moderation model that the provider
has since retired. Leaving that call in place would produce a filter that
returns "allowed" because the request failed, which is worse than shipping no
filter at all: it manufactures confidence.

So the filter is behind a flag, the flag is off, and a disabled filter says so
in its verdict rather than returning a bare ``True``. A caller can then surface
"unfiltered" honestly.

To enable: set ``AGENTIC_RAG_SAFETY_MODEL`` to an available moderation model,
implement :class:`ModerationBackend` against it, and set
``AGENTIC_RAG_ENABLE_SAFETY_FILTER=true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_rag.exceptions import SafetyFilterError
from agentic_rag.settings import Settings, get_settings


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """The outcome of a safety check."""

    allowed: bool
    reason: str

    @property
    def was_screened(self) -> bool:
        """Whether a filter actually ran.

        ``allowed`` alone is ambiguous: it is also what a disabled filter
        returns. Callers that care about the difference check this.
        """
        return self.reason != "filter_disabled"


class ModerationBackend(Protocol):
    """A moderation model."""

    def classify(self, text: str) -> SafetyVerdict:
        """Return a verdict for ``text``."""
        ...


def check_input(
    text: str,
    *,
    settings: Settings | None = None,
    backend: ModerationBackend | None = None,
) -> SafetyVerdict:
    """Screen ``text`` before it reaches the agent."""
    resolved = settings or get_settings()

    if not resolved.enable_safety_filter:
        return SafetyVerdict(allowed=True, reason="filter_disabled")

    if backend is None:
        message = (
            "AGENTIC_RAG_ENABLE_SAFETY_FILTER is true but no moderation backend "
            "is configured. Refusing to fail open."
        )
        raise SafetyFilterError(message)

    return backend.classify(text)
