"""Structural interfaces for every external dependency the agent uses.

These are :class:`typing.Protocol` definitions, not base classes. Nothing has to
inherit from them; a class satisfies one by having the right shape. That is what
lets the graph be assembled from real clients in production and from
deterministic fakes in tests, with no branching inside the agent itself.

The agent depends on these protocols and never on a vendor SDK. Swapping the
model provider or the vector store is an edit to one adapter module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single unit of retrieved context."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """The outcome of a single judge call.

    ``score`` is ``None`` when the judge's reply could not be read as a number
    in ``[0, 1]``. That is a distinct outcome from a low score and the routing
    layer treats it as such: an unreadable verdict is never a pass.
    """

    score: float | None
    raw: str

    @property
    def parsed(self) -> bool:
        """Whether a usable score was extracted."""
        return self.score is not None


@runtime_checkable
class ChatModel(Protocol):
    """A text-in, text-out language model."""

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        """Return the model's completion for ``prompt``."""
        ...


@runtime_checkable
class DocumentRetriever(Protocol):
    """A source of context chunks for a query."""

    def retrieve(self, query: str, *, k: int | None = None) -> Sequence[RetrievedChunk]:
        """Return up to ``k`` chunks relevant to ``query``."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Scores one property of a response."""

    name: str

    def evaluate(self, **fields: str) -> JudgeResult:
        """Score the supplied fields and return the verdict."""
        ...
