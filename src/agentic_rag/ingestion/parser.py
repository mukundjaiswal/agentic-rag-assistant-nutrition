"""Document parsing.

Layout-aware parsing rather than plain text extraction: flattening a table into
prose destroys the row-to-column relationships that an answer depends on, and no
amount of downstream prompting recovers them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from agentic_rag.exceptions import IngestionError
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".txt"})


def iter_source_files(source: Path) -> Iterator[Path]:
    """Yield every supported file under ``source``, sorted for reproducibility."""
    if not source.exists():
        message = f"Source path does not exist: {source}"
        raise IngestionError(message)
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


class DocumentParser:
    """Parses source files into markdown text."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Hold settings; the vendor parser is built lazily on first PDF."""
        self._settings = settings or get_settings()
        self._parser: object | None = None

    def _pdf_parser(self) -> object:
        if self._parser is None:
            from llama_parse import LlamaParse

            self._parser = LlamaParse(
                api_key=self._settings.require_llama_cloud_api_key(),
                result_type="markdown",
            )
        return self._parser

    def parse(self, path: Path) -> str:
        """Return the text content of ``path`` as markdown."""
        if path.suffix.lower() in {".md", ".txt"}:
            return path.read_text(encoding="utf-8")

        documents = self._pdf_parser().load_data(str(path))  # type: ignore[attr-defined]
        if not documents:
            message = f"Parser returned no content for {path}"
            raise IngestionError(message)
        text = "\n\n".join(getattr(d, "text", str(d)) for d in documents)
        logger.debug("ingest.parsed", extra={"path": str(path), "chars": len(text)})
        return text
