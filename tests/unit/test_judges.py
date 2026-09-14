import pytest

from agentic_rag.agent.judges import TemplateJudge, groundedness_judge, precision_judge
from agentic_rag.exceptions import JudgeParseError
from agentic_rag.llm.fake import ScriptedChatModel


def judge_returning(reply: str) -> TemplateJudge:
    return TemplateJudge("test", "{a}", ScriptedChatModel([reply]))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("0.85", 0.85),
        ("1", 1.0),
        ("0", 0.0),
        (" 0.42 ", 0.42),
        ("Score: 0.6", 0.6),
        (".75", 0.75),
    ],
)
def test_reads_a_score_out_of_the_reply(reply, expected):
    assert judge_returning(reply).evaluate(a="x").score == pytest.approx(expected)


@pytest.mark.parametrize("reply", ["", "not sure", "high", "1.4", "-0.2", "42"])
def test_unreadable_or_out_of_range_replies_yield_none(reply):
    """Out of range is a malfunction, not a clamp-and-continue."""
    result = judge_returning(reply).evaluate(a="x")
    assert result.score is None
    assert result.parsed is False


def test_raw_reply_is_retained_for_debugging():
    result = judge_returning("uhh maybe").evaluate(a="x")
    assert result.raw == "uhh maybe"


def test_strict_mode_raises_where_a_bad_verdict_is_a_bug():
    with pytest.raises(JudgeParseError, match="unparseable"):
        judge_returning("no idea").evaluate_strict(a="x")


def test_strict_mode_returns_the_score_when_readable():
    assert judge_returning("0.9").evaluate_strict(a="x") == pytest.approx(0.9)


def test_groundedness_prompt_carries_context_and_response():
    model = ScriptedChatModel(["0.9"])
    groundedness_judge(model).evaluate(context="CTX", response="RESP")
    assert "CTX" in model.prompts[0]
    assert "RESP" in model.prompts[0]


def test_precision_prompt_carries_question_and_response():
    model = ScriptedChatModel(["0.9"])
    precision_judge(model).evaluate(question="Q", response="RESP")
    assert "Q" in model.prompts[0]
    assert "RESP" in model.prompts[0]
