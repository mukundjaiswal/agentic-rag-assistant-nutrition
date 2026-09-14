import pytest
from pydantic import ValidationError

from agentic_rag.exceptions import ConfigurationError
from agentic_rag.settings import Settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None)
    assert 0.0 <= settings.groundedness_threshold <= 1.0
    assert settings.max_iterations >= 1


@pytest.mark.parametrize("value", [-0.1, 1.1, 42.0])
def test_threshold_outside_unit_interval_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, groundedness_threshold=value)


def test_non_positive_iteration_cap_is_rejected():
    """A cap of zero would make every run terminate before it could repair."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_iterations=0)


def test_blank_collection_name_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, collection_name="   ")


def test_missing_api_key_fails_with_an_actionable_message():
    settings = Settings(_env_file=None, openai_api_key=None)
    with pytest.raises(ConfigurationError, match="AGENTIC_RAG_OPENAI_API_KEY"):
        settings.require_openai_api_key()


def test_api_key_is_not_exposed_by_repr():
    """A settings object ends up in logs and tracebacks; keys must not."""
    settings = Settings(_env_file=None, openai_api_key="sk-not-a-real-key")
    assert "sk-not-a-real-key" not in repr(settings)
    assert settings.require_openai_api_key() == "sk-not-a-real-key"


def test_settings_are_immutable():
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.max_iterations = 9
