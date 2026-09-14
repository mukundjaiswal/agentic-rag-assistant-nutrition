"""Shared fixtures.

Every fixture here is offline. The test suite must run in CI with no API key and
no network, otherwise it stops being run and stops being worth anything.
"""

from __future__ import annotations

import pytest

from agentic_rag.agent.dependencies import AgentDependencies
from agentic_rag.agent.judges import FixedJudge
from agentic_rag.llm.fake import ScriptedChatModel
from agentic_rag.retrieval.fake import InMemoryRetriever
from agentic_rag.settings import Settings

CORPUS = [
    "Iron is absorbed more efficiently in the presence of vitamin C.",
    "Calcium competes with iron for absorption and should be taken separately.",
    "Vitamin D supports calcium absorption in the small intestine.",
    "Excess sodium intake is associated with elevated blood pressure.",
]


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that ignore any .env on the developer's machine."""
    return Settings(
        _env_file=None,
        groundedness_threshold=0.7,
        precision_threshold=0.7,
        max_iterations=3,
        retrieval_k=2,
    )


@pytest.fixture
def retriever() -> InMemoryRetriever:
    """Keyword retriever over a small fixed corpus."""
    return InMemoryRetriever(CORPUS, default_k=2)


@pytest.fixture
def model() -> ScriptedChatModel:
    """Chat model that echoes a marker so prompts can be told apart."""
    return ScriptedChatModel(responder=lambda prompt: f"reply::{len(prompt)}")


def make_deps(
    settings: Settings,
    *,
    model: ScriptedChatModel,
    retriever: InMemoryRetriever,
    groundedness: float | None,
    precision: float | None,
) -> AgentDependencies:
    """Build a dependency set whose judges return fixed verdicts."""
    return AgentDependencies(
        model=model,
        retriever=retriever,
        groundedness=FixedJudge("groundedness", groundedness),
        precision=FixedJudge("precision", precision),
        settings=settings,
    )
