"""Routing decisions, as pure functions.

Deliberately free of models, I/O and state mutation. Routing is the part of an
agent that fails quietly: a misordered condition sends every run down the repair
path and the only symptom is a larger invoice. Pure functions make that a unit
test rather than a production surprise.

Two failure modes, two repairs:

* Low **groundedness** means the answer drifted from the context. The retrieval
  was fine; the generation was not. Repair the response.
* Low **precision** means the answer is faithful to the wrong material. Repair
  the query and retrieve again.
"""

from __future__ import annotations

from agentic_rag.agent.state import AgentState, Verdict
from agentic_rag.settings import Settings


def budget_exhausted(state: AgentState, settings: Settings) -> bool:
    """Whether the run has spent its repair budget."""
    return state.get("iteration", 0) >= settings.max_iterations


def passes(score: float | None, threshold: float) -> bool:
    """Whether ``score`` clears ``threshold``.

    ``None`` never passes. An unreadable judge is treated as a failed check, not
    a skipped one.
    """
    return score is not None and score >= threshold


def route_after_groundedness(state: AgentState, settings: Settings) -> Verdict:
    """Decide what follows the groundedness check."""
    if passes(state.get("groundedness"), settings.groundedness_threshold):
        return Verdict.CHECK_PRECISION
    if budget_exhausted(state, settings):
        return Verdict.EXHAUSTED
    return Verdict.REFINE_RESPONSE


def route_after_precision(state: AgentState, settings: Settings) -> Verdict:
    """Decide what follows the precision check."""
    if passes(state.get("precision"), settings.precision_threshold):
        return Verdict.COMPLETE
    if budget_exhausted(state, settings):
        return Verdict.EXHAUSTED
    return Verdict.REFINE_QUERY
