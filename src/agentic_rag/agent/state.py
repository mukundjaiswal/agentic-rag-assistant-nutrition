"""The state carried between graph nodes.

Kept as a ``TypedDict`` because that is what the graph runtime merges node
return values into. :class:`Verdict` names the routing outcomes so that a typo
in a destination string is a type error rather than a silently unreachable node.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict


class Verdict(StrEnum):
    """Where routing sends a run next."""

    CHECK_PRECISION = "check_precision"
    REFINE_RESPONSE = "refine_response"
    REFINE_QUERY = "refine_query"
    EXHAUSTED = "max_iterations_reached"
    COMPLETE = "complete"


class AgentState(TypedDict, total=False):
    """Everything one run accumulates.

    Attributes:
        run_id: Correlates log lines belonging to this run.
        question: The user's question, unmodified.
        query: The search query, which expansion and refinement rewrite.
        context: Retrieved context, joined.
        response: The current candidate answer.
        groundedness: Support for the response in the context; ``None`` if the
            judge's reply could not be read.
        precision: How directly the response answers the question.
        iteration: Repairs performed so far, against the configured cap.
        exhausted: Set when the run stopped at the cap without passing.
        history: Ordered node names, for tracing what actually happened.
    """

    run_id: str
    question: str
    query: str
    context: str
    response: str
    groundedness: float | None
    precision: float | None
    iteration: int
    exhausted: bool
    history: list[str]


def initial_state(question: str, *, run_id: str) -> AgentState:
    """Build the starting state for ``question``."""
    return AgentState(
        run_id=run_id,
        question=question,
        query=question,
        iteration=0,
        exhausted=False,
        history=[],
    )
