"""The metrics this agent is judged on.

Registered with the shared harness rather than defined inside it. Groundedness,
precision and the repair budget are properties of *this* agent; a classifier has
no use for them, and a harness that shipped them would be pretending otherwise.

Registration is an explicit call, not an import side effect. Importing this
module does nothing; :func:`register_agent_metrics` is what mutates the shared
registry. That matters because the registry is global and shared with sibling
projects: a module that registers itself on import means any transitive import
silently changes what every other consumer sees. Call it once, where a run is
being set up.
"""

from __future__ import annotations

from typing import Final

from eval_harness import Direction, mean_of, none_rate_of, rate_of
from eval_harness.metrics import register_metric

MEAN_GROUNDEDNESS: Final = "mean_groundedness"
MEAN_PRECISION: Final = "mean_precision"
MEAN_REPAIRS: Final = "mean_repairs"
EXHAUSTION_RATE: Final = "exhaustion_rate"
UNPARSED_JUDGE_RATE: Final = "unparsed_judge_rate"

AGENT_METRICS: Final = (
    MEAN_GROUNDEDNESS,
    MEAN_PRECISION,
    MEAN_REPAIRS,
    EXHAUSTION_RATE,
    UNPARSED_JUDGE_RATE,
    "p50_latency_seconds",
    "p95_latency_seconds",
)

_REGISTERED = False


def register_agent_metrics(*, replace: bool = False) -> tuple[str, ...]:
    """Register this agent's metrics and return their names.

    Idempotent, so importing it from both the CLI and a notebook is safe.
    """
    global _REGISTERED
    if _REGISTERED and not replace:
        return AGENT_METRICS

    register_metric(
        MEAN_GROUNDEDNESS,
        mean_of("groundedness"),
        direction=Direction.HIGHER_IS_BETTER,
        description="How well answers are supported by the retrieved context.",
        replace=True,
    )
    register_metric(
        MEAN_PRECISION,
        mean_of("precision"),
        direction=Direction.HIGHER_IS_BETTER,
        description="How directly answers address the question asked.",
        replace=True,
    )
    register_metric(
        MEAN_REPAIRS,
        mean_of("iteration"),
        direction=Direction.LOWER_IS_BETTER,
        description=(
            "Repairs per question. A leading indicator: this rises before "
            "answer quality falls, because the loop absorbs the damage first."
        ),
        replace=True,
    )
    register_metric(
        EXHAUSTION_RATE,
        rate_of("exhausted"),
        direction=Direction.LOWER_IS_BETTER,
        description="Share of questions the agent declined at the repair budget.",
        replace=True,
    )
    register_metric(
        UNPARSED_JUDGE_RATE,
        none_rate_of("groundedness"),
        direction=Direction.LOWER_IS_BETTER,
        description=(
            "Share of runs where a judge's verdict could not be read. Harness "
            "health, not answer quality: if non-zero, every quality number "
            "above it is measured on a smaller sample than the case count."
        ),
        replace=True,
    )

    _REGISTERED = True
    return AGENT_METRICS
