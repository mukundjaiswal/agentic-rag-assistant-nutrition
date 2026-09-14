"""Index-time augmentation.

Users ask questions; documents make statements. Embedding a passage alongside
questions that passage answers narrows that phrasing gap, and it is cheaper to
close at index time than to keep paying for it at query time.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.ingestion.chunker import Chunk
from agentic_rag.interfaces import ChatModel

HYPOTHETICAL_QUESTIONS_PROMPT = """\
Write {n} questions that the passage below answers directly.
One question per line. No numbering, no preamble.

Passage:
{passage}
"""


def with_hypothetical_questions(
    chunks: Sequence[Chunk], model: ChatModel, *, n: int = 3
) -> list[Chunk]:
    """Return ``chunks`` with generated questions appended to each body."""
    augmented: list[Chunk] = []
    for chunk in chunks:
        questions = model.complete(
            HYPOTHETICAL_QUESTIONS_PROMPT.format(n=n, passage=chunk.text)
        ).strip()
        metadata = {**chunk.metadata, "hypothetical_questions": questions}
        body = (
            f"{chunk.text}\n\nRelated questions:\n{questions}"
            if questions
            else chunk.text
        )
        augmented.append(Chunk(text=body, metadata=metadata))
    return augmented
