"""The ingestion pipeline, assembled from its stages.

Each stage is independently testable; this module only sequences them and
reports what happened. Ingestion runs are the thing you have to explain when
retrieval quality changes, so the report is part of the output, not a log line.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.ingestion.augment import with_hypothetical_questions
from agentic_rag.ingestion.chunker import Chunk, semantic_chunks
from agentic_rag.ingestion.parser import DocumentParser, iter_source_files
from agentic_rag.interfaces import ChatModel
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings

logger = get_logger(__name__)

Chunker = Callable[[str], list[Chunk]]


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """What an ingestion run produced."""

    files_parsed: int
    chunks_created: int
    chunks_indexed: int
    augmented: bool

    def summary(self) -> str:
        """One-line human-readable summary."""
        suffix = " (augmented)" if self.augmented else ""
        return (
            f"{self.files_parsed} file(s) -> {self.chunks_created} chunk(s) -> "
            f"{self.chunks_indexed} indexed{suffix}"
        )


class IngestionPipeline:
    """Parses, chunks, optionally augments, and indexes a corpus."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        parser: DocumentParser | None = None,
        chunker: Chunker | None = None,
        model: ChatModel | None = None,
        store: Any | None = None,
    ) -> None:
        """Wire the stages, defaulting each to the production implementation."""
        self._settings = settings or get_settings()
        self._parser = parser or DocumentParser(self._settings)
        self._chunker = chunker or (lambda text: semantic_chunks(text, self._settings))
        self._model = model
        self._store = store

    def run(self, source: Path, *, augment: bool = True) -> IngestionReport:
        """Ingest every supported file under ``source``."""
        chunks: list[Chunk] = []
        files = 0

        for path in iter_source_files(source):
            files += 1
            text = self._parser.parse(path)
            for chunk in self._chunker(text):
                chunks.append(
                    Chunk(
                        text=chunk.text,
                        metadata={**chunk.metadata, "source": str(path)},
                    )
                )

        created = len(chunks)
        did_augment = False
        if augment and self._model is not None and chunks:
            chunks = with_hypothetical_questions(chunks, self._model)
            did_augment = True

        indexed = self._index(chunks)
        report = IngestionReport(files, created, indexed, did_augment)
        logger.info("ingest.complete", extra={"report": report.summary()})
        return report

    def _index(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        store = self._store
        if store is None:
            from agentic_rag.retrieval.vectorstore import get_vectorstore

            store = get_vectorstore()
        store.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)
