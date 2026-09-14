import pytest

from agentic_rag.agent.routing import (
    budget_exhausted,
    passes,
    route_after_groundedness,
    route_after_precision,
)
from agentic_rag.agent.state import Verdict


def test_grounded_response_advances_to_the_precision_check(settings):
    state = {"groundedness": 0.95, "iteration": 0}
    assert route_after_groundedness(state, settings) is Verdict.CHECK_PRECISION


def test_ungrounded_response_is_repaired(settings):
    state = {"groundedness": 0.2, "iteration": 0}
    assert route_after_groundedness(state, settings) is Verdict.REFINE_RESPONSE


def test_unreadable_verdict_does_not_pass(settings):
    """A gate that fails open is not a gate."""
    state = {"groundedness": None, "iteration": 0}
    assert route_after_groundedness(state, settings) is Verdict.REFINE_RESPONSE


def test_score_exactly_at_the_threshold_passes(settings):
    state = {"groundedness": settings.groundedness_threshold, "iteration": 0}
    assert route_after_groundedness(state, settings) is Verdict.CHECK_PRECISION


def test_a_passing_score_is_honoured_even_at_the_budget_ceiling(settings):
    """Spending the budget must not discard an answer that already passed."""
    state = {"groundedness": 0.99, "iteration": settings.max_iterations}
    assert route_after_groundedness(state, settings) is Verdict.CHECK_PRECISION


def test_budget_ceiling_stops_a_failing_run(settings):
    state = {"groundedness": 0.1, "iteration": settings.max_iterations}
    assert route_after_groundedness(state, settings) is Verdict.EXHAUSTED


def test_precise_response_completes(settings):
    assert route_after_precision({"precision": 0.9, "iteration": 1}, settings) is (
        Verdict.COMPLETE
    )


def test_imprecise_response_repairs_the_query_not_the_response(settings):
    """Faithful to the wrong material means retrieve again, not rewrite."""
    assert route_after_precision({"precision": 0.3, "iteration": 1}, settings) is (
        Verdict.REFINE_QUERY
    )


def test_precision_run_stops_at_the_budget_ceiling(settings):
    state = {"precision": 0.3, "iteration": settings.max_iterations}
    assert route_after_precision(state, settings) is Verdict.EXHAUSTED


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [(1.0, 0.7, True), (0.7, 0.7, True), (0.69, 0.7, False), (None, 0.0, False)],
)
def test_passes_treats_none_as_failure(score, threshold, expected):
    assert passes(score, threshold) is expected


def test_budget_exhausted_is_inclusive(settings):
    assert budget_exhausted({"iteration": settings.max_iterations}, settings)
    assert not budget_exhausted({"iteration": settings.max_iterations - 1}, settings)


def test_missing_iteration_key_is_treated_as_zero(settings):
    assert not budget_exhausted({}, settings)
