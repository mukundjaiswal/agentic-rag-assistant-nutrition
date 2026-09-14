"""Chroma-backed retrieval.

Chroma was chosen because it runs from a local directory with no service to
stand up. That is the right trade for a single-node assistant and the wrong one
the moment the index has to be shared across processes; the
:class:`~agentic_rag.interfaces.DocumentRetriever` protocol is what keeps that
future swap to one module.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from agentic_rag.exceptions import RetrievalError
from agentic_rag.interfaces import RetrievedChunk
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_vectorstore() -> Any:
    """Return the process-wide Chroma collection."""
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=OpenAIEmbeddings(api_key=settings.require_openai_api_key()),
    )


class ChromaRetriever:
    """Similarity retrieval over the Chroma collection."""

    def __init__(
        self, settings: Settings | None = None, *, store: Any | None = None
    ) -> None:
        """Bind to a collection, defaulting to the configured one."""
        self._settings = settings or get_settings()
        self._store = store if store is not None else get_vectorstore()

    def retrieve(self, query: str, *, k: int | None = None) -> Sequence[RetrievedChunk]:
        """Return up to ``k`` chunks relevant to ``query``."""
        limit = k or self._settings.retrieval_k
        # Broad on purpose: vendor clients raise their own unrelated exception
        # types, and callers should only ever have to catch RetrievalError.
        try:
            hits = self._store.similarity_search_with_relevance_scores(query, k=limit)
        except Exception as exc:
            message = f"Retrieval failed for query {query!r}"
            raise RetrievalError(message) from exc

        chunks = [
            RetrievedChunk(
                text=document.page_content,
                metadata=dict(document.metadata or {}),
                score=float(score),
            )
            for document, score in hits
        ]
        logger.debug("retrieval.hits", extra={"query": query, "n": len(chunks)})
        return chunks


def build_self_query_retriever(
    metadata_field_info: Sequence[Any], document_description: str
) -> Any:
    """Build a retriever that derives metadata filters from the question.

    Useful when the corpus carries structured metadata and a question implies a
    filter without stating one ("what changed in the 2024 guidance") . Returned
    as the vendor object because the self-query chain owns its own call
    sequence; wrap it before handing it to the agent.
    """
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return SelfQueryRetriever.from_llm(
        ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.require_openai_api_key(),
        ),
        get_vectorstore(),
        document_description,
        list(metadata_field_info),
    )
