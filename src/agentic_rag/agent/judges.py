"""LLM-as-judge scoring.

This is the layer that decides whether a response ships, which is what separates
this from a retrieve-and-generate demo.

Two design commitments:

1. An unreadable verdict is not a pass. :meth:`_parse` returns ``None`` and the
   routing layer sends the run to repair. A quality gate that fails open is not
   a gate.
2. Judges are ordinary objects satisfying :class:`~agentic_rag.interfaces.Judge`,
   so a test can substitute a fixed verdict and assert on routing alone.

Known limitation, stated rather than buried: these judges are not calibrated
against human labels. Until agreement is measured, a score reports the judge's
own consistency, not correctness.
"""

from __future__ import annotations

import re
from typing import Final

from agentic_rag.agent.prompts import GROUNDEDNESS_TEMPLATE, PRECISION_TEMPLATE
from agentic_rag.exceptions import JudgeParseError
from agentic_rag.interfaces import ChatModel, JudgeResult
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)

# The sign is captured deliberately. Without it, a judge replying "-0.2" parses
# as 0.2 and a malfunctioning judge reads as a plausible low score.
_NUMBER: Final = re.compile(r"-?\d*\.?\d+")


class TemplateJudge:
    """Scores one property by filling a template and parsing a bare number."""

    def __init__(self, name: str, template: str, model: ChatModel) -> None:
        """Bind a scoring template to a model."""
        self.name = name
        self._template = template
        self._model = model

    def evaluate(self, **fields: str) -> JudgeResult:
        """Score ``fields`` and return the verdict, parsed or not."""
        raw = self._model.complete(self._template.format(**fields), temperature=0.0)
        score = self._parse(raw)
        if score is None:
            logger.warning(
                "judge.unparsed", extra={"judge": self.name, "raw": raw[:200]}
            )
        return JudgeResult(score=score, raw=raw)

    def evaluate_strict(self, **fields: str) -> float:
        """Score ``fields``, raising if the verdict cannot be read.

        For offline evaluation, where an unreadable judge is a bug to fix rather
        than a condition to route around.
        """
        result = self.evaluate(**fields)
        if result.score is None:
            message = f"{self.name} returned an unparseable verdict: {result.raw!r}"
            raise JudgeParseError(message)
        return result.score

    @staticmethod
    def _parse(text: str) -> float | None:
        match = _NUMBER.search(text or "")
        if match is None:
            return None
        try:
            value = float(match.group())
        except ValueError:
            return None
        return value if 0.0 <= value <= 1.0 else None


def groundedness_judge(model: ChatModel) -> TemplateJudge:
    """Judge of whether the response is supported by the retrieved context."""
    return TemplateJudge("groundedness", GROUNDEDNESS_TEMPLATE, model)


def precision_judge(model: ChatModel) -> TemplateJudge:
    """Judge of whether the response answers the question that was asked."""
    return TemplateJudge("precision", PRECISION_TEMPLATE, model)


class FixedJudge:
    """Returns a preset verdict. For tests of routing behaviour."""

    def __init__(self, name: str, score: float | None, raw: str = "") -> None:
        """Always return ``score``."""
        self.name = name
        self._score = score
        self._raw = raw or str(score)

    def evaluate(self, **fields: str) -> JudgeResult:
        """Return the preset verdict, ignoring the fields."""
        del fields
        return JudgeResult(score=self._score, raw=self._raw)
