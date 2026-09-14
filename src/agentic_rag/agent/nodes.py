"""Graph nodes.

One responsibility each. Every node takes the state and returns only the keys it
changed, which keeps merge semantics obvious and makes a node testable in
isolation.
"""

from __future__ import annotations

from typing import Any

from agentic_rag.agent.dependencies import AgentDependencies
from agentic_rag.agent.prompts import (
    ANSWER_TEMPLATE,
    EXHAUSTED_RESPONSE,
    QUERY_EXPANSION_TEMPLATE,
    QUERY_REFINEMENT_TEMPLATE,
    RESPONSE_REFINEMENT_TEMPLATE,
    RESPONSE_SYSTEM_MESSAGE,
)
from agentic_rag.agent.state import AgentState
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)

CONTEXT_SEPARATOR = "\n\n---\n\n"


class AgentNodes:
    """The node implementations, bound to a dependency set."""

    def __init__(self, deps: AgentDependencies) -> None:
        """Bind collaborators."""
        self._deps = deps

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _trace(state: AgentState, node: str) -> list[str]:
        return [*state.get("history", []), node]

    def _log(self, state: AgentState, event: str, **fields: Any) -> None:
        logger.info(event, extra={"run_id": state.get("run_id", "-"), **fields})

    # --- nodes --------------------------------------------------------------

    def expand_query(self, state: AgentState) -> dict[str, Any]:
        """Rewrite the question into a retrieval-friendly query."""
        query = self._deps.model.complete(
            QUERY_EXPANSION_TEMPLATE.format(question=state["question"])
        ).strip()
        self._log(state, "node.expand_query")
        return {
            "query": query or state["question"],
            "history": self._trace(state, "expand_query"),
        }

    def retrieve_context(self, state: AgentState) -> dict[str, Any]:
        """Fetch context for the current query."""
        chunks = self._deps.retriever.retrieve(state["query"])
        self._log(state, "node.retrieve_context", chunks=len(chunks))
        return {
            "context": CONTEXT_SEPARATOR.join(chunk.text for chunk in chunks),
            "history": self._trace(state, "retrieve_context"),
        }

    def craft_response(self, state: AgentState) -> dict[str, Any]:
        """Answer the question from the retrieved context."""
        response = self._deps.model.complete(
            ANSWER_TEMPLATE.format(
                system_message=RESPONSE_SYSTEM_MESSAGE,
                context=state.get("context", ""),
                question=state["question"],
            )
        ).strip()
        self._log(state, "node.craft_response")
        return {"response": response, "history": self._trace(state, "craft_response")}

    def score_groundedness(self, state: AgentState) -> dict[str, Any]:
        """Judge whether the response is supported by the context."""
        verdict = self._deps.groundedness.evaluate(
            context=state.get("context", ""), response=state.get("response", "")
        )
        self._log(state, "node.score_groundedness", score=verdict.score)
        return {
            "groundedness": verdict.score,
            "history": self._trace(state, "score_groundedness"),
        }

    def check_precision(self, state: AgentState) -> dict[str, Any]:
        """Judge whether the response answers the question asked."""
        verdict = self._deps.precision.evaluate(
            question=state["question"], response=state.get("response", "")
        )
        self._log(state, "node.check_precision", score=verdict.score)
        return {
            "precision": verdict.score,
            "history": self._trace(state, "check_precision"),
        }

    def refine_response(self, state: AgentState) -> dict[str, Any]:
        """Repair a response that drifted from otherwise adequate context."""
        response = self._deps.model.complete(
            RESPONSE_REFINEMENT_TEMPLATE.format(
                context=state.get("context", ""), response=state.get("response", "")
            )
        ).strip()
        self._log(
            state, "node.refine_response", iteration=state.get("iteration", 0) + 1
        )
        return {
            "response": response,
            "iteration": state.get("iteration", 0) + 1,
            "history": self._trace(state, "refine_response"),
        }

    def refine_query(self, state: AgentState) -> dict[str, Any]:
        """Repair a query that retrieved the wrong material."""
        query = self._deps.model.complete(
            QUERY_REFINEMENT_TEMPLATE.format(
                question=state["question"], query=state.get("query", "")
            )
        ).strip()
        self._log(state, "node.refine_query", iteration=state.get("iteration", 0) + 1)
        return {
            "query": query or state.get("query", state["question"]),
            "iteration": state.get("iteration", 0) + 1,
            "history": self._trace(state, "refine_query"),
        }

    def max_iterations_reached(self, state: AgentState) -> dict[str, Any]:
        """Terminate honestly instead of spending another repair.

        A self-correcting loop without a cap is an unbounded cost and latency
        risk. In a support setting, "I could not answer this confidently" is a
        better outcome than a fourth rewrite of a wrong answer.
        """
        self._log(state, "node.exhausted", iterations=state.get("iteration", 0))
        return {
            "response": EXHAUSTED_RESPONSE,
            "exhausted": True,
            "history": self._trace(state, "max_iterations_reached"),
        }
