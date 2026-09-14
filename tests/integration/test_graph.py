"""End-to-end runs of the compiled graph, driven entirely by fakes.

Skipped when the ``rag`` extra is not installed, so a contributor can run the
unit suite without pulling the full dependency tree. CI installs it.
"""

from __future__ import annotations

import pytest

from agentic_rag.agent.graph import AgentRunner
from tests.conftest import make_deps

pytest.importorskip("langgraph", reason="install the 'rag' extra to run these")

pytestmark = pytest.mark.integration


def runner(settings, model, retriever, *, groundedness, precision) -> AgentRunner:
    return AgentRunner(
        make_deps(
            settings,
            model=model,
            retriever=retriever,
            groundedness=groundedness,
            precision=precision,
        )
    )


def test_a_passing_run_returns_without_repairing(settings, model, retriever):
    state = runner(settings, model, retriever, groundedness=0.95, precision=0.95).ask(
        "How is iron absorbed?"
    )

    assert state["iteration"] == 0
    assert state.get("exhausted", False) is False
    assert "refine_response" not in state["history"]
    assert "refine_query" not in state["history"]


def test_a_failing_run_stops_at_the_budget_and_says_so(settings, model, retriever):
    state = runner(settings, model, retriever, groundedness=0.1, precision=0.1).ask(
        "How is iron absorbed?"
    )

    assert state["exhausted"] is True
    assert state["iteration"] == settings.max_iterations
    assert "could not answer this confidently" in state["response"]


def test_low_groundedness_repairs_the_response_not_the_query(
    settings, model, retriever
):
    state = runner(settings, model, retriever, groundedness=0.1, precision=0.95).ask(
        "How is iron absorbed?"
    )

    assert "refine_response" in state["history"]
    assert "refine_query" not in state["history"]


def test_low_precision_repairs_the_query_and_retrieves_again(
    settings, model, retriever
):
    state = runner(settings, model, retriever, groundedness=0.95, precision=0.1).ask(
        "How is iron absorbed?"
    )

    assert "refine_query" in state["history"]
    assert state["history"].count("retrieve_context") > 1


def test_an_unreadable_judge_never_silently_passes(settings, model, retriever):
    state = runner(settings, model, retriever, groundedness=None, precision=0.95).ask(
        "How is iron absorbed?"
    )

    assert state["exhausted"] is True


def test_every_run_carries_a_correlation_id(settings, model, retriever):
    state = runner(settings, model, retriever, groundedness=0.95, precision=0.95).ask(
        "How is iron absorbed?", run_id="fixed-id"
    )
    assert state["run_id"] == "fixed-id"
