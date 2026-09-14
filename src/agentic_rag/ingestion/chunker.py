"""Chunking.

Semantic chunking rather than a fixed character window, so that a boundary does
not land in the middle of an argument and leave both halves unanswerable.

A character-count fallback is provided because semantic chunking costs embedding
calls, and a smoke test of the pipeline should not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.settings import Settings, get_settings


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of indexable text with its provenance."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def fixed_size_chunks(
    text: str, *, size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Split ``text`` on a sliding character window.

    Deterministic and free. Used for smoke tests and as the fallback when
    embedding-based chunking is unavailable.
    """
    if size <= 0:
        message = "size must be positive"
        raise ValueError(message)
    if not 0 <= overlap < size:
        message = "overlap must be non-negative and smaller than size"
        raise ValueError(message)

    chunks: list[Chunk] = []
    step = size - overlap
    for start in range(0, max(len(text), 1), step):
        window = text[start : start + size].strip()
        if window:
            chunks.append(Chunk(text=window, metadata={"offset": start}))
        if start + size >= len(text):
            break
    return chunks


def semantic_chunks(text: str, settings: Settings | None = None) -> list[Chunk]:
    """Split ``text`` on semantic boundaries using an embedding model."""
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    resolved = settings or get_settings()
    splitter = SemanticChunker(
        OpenAIEmbeddings(api_key=resolved.require_openai_api_key())
    )
    documents: Sequence[Any] = splitter.create_documents([text])
    return [
        Chunk(text=d.page_content, metadata=dict(d.metadata or {})) for d in documents
    ]
