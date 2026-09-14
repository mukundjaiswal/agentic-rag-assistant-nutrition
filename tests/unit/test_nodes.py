from agentic_rag.agent.nodes import AgentNodes
from agentic_rag.agent.prompts import EXHAUSTED_RESPONSE
from tests.conftest import make_deps


def build(settings, model, retriever):
    return AgentNodes(
        make_deps(
            settings,
            model=model,
            retriever=retriever,
            groundedness=0.9,
            precision=0.9,
        )
    )


def test_retrieve_joins_chunks_into_context(settings, model, retriever):
    nodes = build(settings, model, retriever)
    result = nodes.retrieve_context({"question": "iron", "query": "iron absorption"})
    assert result["context"]
    assert retriever.queries == ["iron absorption"]


def test_expand_query_falls_back_when_the_model_returns_nothing(settings, retriever):
    from agentic_rag.llm.fake import ScriptedChatModel

    nodes = build(settings, ScriptedChatModel([""]), retriever)
    result = nodes.expand_query({"question": "iron?", "query": "iron?"})
    assert result["query"] == "iron?"


def test_refine_response_increments_the_repair_count(settings, model, retriever):
    nodes = build(settings, model, retriever)
    result = nodes.refine_response(
        {"question": "q", "context": "c", "response": "r", "iteration": 1}
    )
    assert result["iteration"] == 2


def test_refine_query_increments_the_repair_count(settings, model, retriever):
    nodes = build(settings, model, retriever)
    result = nodes.refine_query({"question": "q", "query": "q", "iteration": 0})
    assert result["iteration"] == 1


def test_exhausted_node_declines_rather_than_guessing(settings, model, retriever):
    nodes = build(settings, model, retriever)
    result = nodes.max_iterations_reached({"question": "q", "iteration": 3})
    assert result["response"] == EXHAUSTED_RESPONSE
    assert result["exhausted"] is True


def test_every_node_appends_to_the_trace(settings, model, retriever):
    nodes = build(settings, model, retriever)
    state = {"question": "q", "query": "q", "history": ["expand_query"]}
    assert nodes.retrieve_context(state)["history"] == [
        "expand_query",
        "retrieve_context",
    ]
