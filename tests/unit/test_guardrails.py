import pytest

from agentic_rag.exceptions import SafetyFilterError
from agentic_rag.safety.guardrails import SafetyVerdict, check_input
from agentic_rag.settings import Settings


class AlwaysBlocks:
    def classify(self, text: str) -> SafetyVerdict:
        del text
        return SafetyVerdict(allowed=False, reason="policy")


def test_disabled_filter_allows_but_says_it_did_not_screen():
    """`allowed` alone is ambiguous; callers need to know nothing ran."""
    settings = Settings(_env_file=None, enable_safety_filter=False)
    verdict = check_input("anything", settings=settings)
    assert verdict.allowed is True
    assert verdict.was_screened is False


def test_enabling_without_a_backend_refuses_to_fail_open():
    settings = Settings(_env_file=None, enable_safety_filter=True)
    with pytest.raises(SafetyFilterError, match="fail open"):
        check_input("anything", settings=settings)


def test_configured_backend_decides():
    settings = Settings(_env_file=None, enable_safety_filter=True)
    verdict = check_input("anything", settings=settings, backend=AlwaysBlocks())
    assert verdict.allowed is False
    assert verdict.was_screened is True
