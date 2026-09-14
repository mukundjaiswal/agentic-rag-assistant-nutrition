"""In-memory retriever for tests.

Keyword overlap, not embeddings. It exists to make retrieval deterministic so
that a test failure points at the agent rather than at embedding drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.interfaces import RetrievedChunk


class InMemoryRetriever:
    """Ranks a fixed corpus by token overlap with the query."""

    def __init__(self, corpus: Sequence[str], *, default_k: int = 5) -> None:
        """Hold ``corpus`` in memory and rank against it."""
        self._corpus = list(corpus)
        self._default_k = default_k
        self.queries: list[str] = []

    def retrieve(self, query: str, *, k: int | None = None) -> Sequence[RetrievedChunk]:
        """Return the ``k`` chunks sharing the most tokens with ``query``."""
        self.queries.append(query)
        limit = k or self._default_k
        wanted = set(query.lower().split())

        def overlap(text: str) -> int:
            return len(wanted & set(text.lower().split()))

        ranked = sorted(self._corpus, key=overlap, reverse=True)
        return [
            RetrievedChunk(
                text=text, metadata={"rank": index}, score=float(overlap(text))
            )
            for index, text in enumerate(ranked[:limit])
        ]
