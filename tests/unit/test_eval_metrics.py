import pytest
from eval_harness import Direction
from eval_harness.metrics import get_metric

from agentic_rag.eval_metrics import (
    EXHAUSTION_RATE,
    MEAN_GROUNDEDNESS,
    MEAN_REPAIRS,
    UNPARSED_JUDGE_RATE,
    register_agent_metrics,
)


@pytest.fixture(autouse=True)
def _registered():
    register_agent_metrics(replace=True)


def test_registration_is_idempotent():
    """Imported from the CLI and from a notebook in the same process."""
    first = register_agent_metrics()
    second = register_agent_metrics()
    assert first == second


def test_quality_metrics_are_higher_is_better():
    assert get_metric(MEAN_GROUNDEDNESS).direction is Direction.HIGHER_IS_BETTER


def test_cost_metrics_are_lower_is_better():
    """More repairs per question is worse, even though the number goes up."""
    assert get_metric(MEAN_REPAIRS).direction is Direction.LOWER_IS_BETTER
    assert get_metric(EXHAUSTION_RATE).direction is Direction.LOWER_IS_BETTER


def test_groundedness_ignores_unreadable_verdicts():
    metric = get_metric(MEAN_GROUNDEDNESS)
    observations = [
        {"groundedness": 1.0},
        {"groundedness": None},
        {"groundedness": 0.0},
    ]
    assert metric(observations) == pytest.approx(0.5)


def test_unparsed_rate_counts_exactly_those_unreadable_verdicts():
    metric = get_metric(UNPARSED_JUDGE_RATE)
    observations = [{"groundedness": None}, {"groundedness": 0.9}]
    assert metric(observations) == pytest.approx(0.5)


def test_exhaustion_rate_counts_declined_questions():
    metric = get_metric(EXHAUSTION_RATE)
    assert metric([{"exhausted": True}, {"exhausted": False}]) == pytest.approx(0.5)


def test_every_declared_metric_is_actually_registered():
    for name in register_agent_metrics():
        assert get_metric(name) is not None


def test_importing_the_package_does_not_mutate_the_shared_registry():
    """The registry is global and shared with sibling projects.

    A module that registered itself on import would mean any transitive import
    silently changed what every other consumer sees.
    """
    import subprocess
    import sys

    probe = (
        "import agentic_rag;"
        "from eval_harness.metrics import registered_metrics;"
        "print(','.join(registered_metrics()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "p50_latency_seconds,p95_latency_seconds"
