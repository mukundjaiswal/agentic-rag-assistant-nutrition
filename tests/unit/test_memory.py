import pytest

from agentic_rag.memory.session import SessionMemory


def test_records_turns_in_order():
    memory = SessionMemory(session_id="s")
    memory.add("q1", "a1")
    memory.add("q2", "a2")
    assert [t.question for t in memory.turns] == ["q1", "q2"]


def test_window_evicts_the_oldest_turn():
    """An unbounded transcript eventually crowds out the retrieved context."""
    memory = SessionMemory(session_id="s", max_turns=2)
    for index in range(5):
        memory.add(f"q{index}", f"a{index}")
    assert len(memory) == 2
    assert [t.question for t in memory.turns] == ["q3", "q4"]


def test_renders_prompt_ready_context():
    memory = SessionMemory(session_id="s")
    memory.add("q", "a")
    assert memory.as_context() == "Q: q\nA: a"


def test_clear_empties_the_window():
    memory = SessionMemory(session_id="s")
    memory.add("q", "a")
    memory.clear()
    assert len(memory) == 0
    assert memory.as_context() == ""


def test_window_size_must_be_positive():
    with pytest.raises(ValueError, match="max_turns"):
        SessionMemory(session_id="s", max_turns=0)
