"""Everything the agent needs from the outside world, in one object.

Constructing this is the only place vendor clients are created. The nodes
receive it and never import an SDK, which is what makes the whole agent
runnable in tests against fakes.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.judges import groundedness_judge, precision_judge
from agentic_rag.interfaces import ChatModel, DocumentRetriever, Judge
from agentic_rag.settings import Settings, get_settings


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    """Collaborators injected into the graph nodes."""

    model: ChatModel
    retriever: DocumentRetriever
    groundedness: Judge
    precision: Judge
    settings: Settings


def build_default_dependencies(settings: Settings | None = None) -> AgentDependencies:
    """Assemble the production dependency set.

    Imports the concrete adapters here rather than at module scope so that
    importing the agent package never requires credentials or vendor packages.
    """
    from agentic_rag.llm.openai_client import OpenAIChatModel
    from agentic_rag.retrieval.vectorstore import ChromaRetriever

    resolved = settings or get_settings()
    model = OpenAIChatModel(resolved)
    return AgentDependencies(
        model=model,
        retriever=ChromaRetriever(resolved),
        groundedness=groundedness_judge(model),
        precision=precision_judge(model),
        settings=resolved,
    )
