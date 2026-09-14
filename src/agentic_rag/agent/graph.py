"""Graph assembly and the runner that drives it.

```
        expand_query
             |
        retrieve_context  <--------------------------+
             |                                       |
        craft_response                               |
             |                                       |
      score_groundedness <-----------+          refine_query
        |            |               |               ^
      pass          fail        refine_response      |
        |            |               |               |
        v            +---------------+               |
      check_precision ------------------- fail ------+
             |
           pass -> END        (budget spent -> max_iterations_reached -> END)
```
"""

from __future__ import annotations

import uuid
from typing import Any

from agentic_rag.agent.dependencies import AgentDependencies, build_default_dependencies
from agentic_rag.agent.nodes import AgentNodes
from agentic_rag.agent.routing import route_after_groundedness, route_after_precision
from agentic_rag.agent.state import AgentState, Verdict, initial_state
from agentic_rag.settings import Settings


def build_graph(deps: AgentDependencies) -> Any:
    """Compile the agent graph against ``deps``."""
    from langgraph.graph import END, StateGraph

    nodes = AgentNodes(deps)
    settings = deps.settings
    graph: Any = StateGraph(AgentState)

    graph.add_node(Verdict.CHECK_PRECISION.value, nodes.check_precision)
    graph.add_node(Verdict.REFINE_RESPONSE.value, nodes.refine_response)
    graph.add_node(Verdict.REFINE_QUERY.value, nodes.refine_query)
    graph.add_node(Verdict.EXHAUSTED.value, nodes.max_iterations_reached)
    graph.add_node("expand_query", nodes.expand_query)
    graph.add_node("retrieve_context", nodes.retrieve_context)
    graph.add_node("craft_response", nodes.craft_response)
    graph.add_node("score_groundedness", nodes.score_groundedness)

    graph.set_entry_point("expand_query")
    graph.add_edge("expand_query", "retrieve_context")
    graph.add_edge("retrieve_context", "craft_response")
    graph.add_edge("craft_response", "score_groundedness")

    graph.add_conditional_edges(
        "score_groundedness",
        lambda state: route_after_groundedness(state, settings).value,
        {
            Verdict.CHECK_PRECISION.value: Verdict.CHECK_PRECISION.value,
            Verdict.REFINE_RESPONSE.value: Verdict.REFINE_RESPONSE.value,
            Verdict.EXHAUSTED.value: Verdict.EXHAUSTED.value,
        },
    )
    graph.add_conditional_edges(
        Verdict.CHECK_PRECISION.value,
        lambda state: route_after_precision(state, settings).value,
        {
            Verdict.COMPLETE.value: END,
            Verdict.REFINE_QUERY.value: Verdict.REFINE_QUERY.value,
            Verdict.EXHAUSTED.value: Verdict.EXHAUSTED.value,
        },
    )

    # A repaired response is re-judged; a repaired query is re-retrieved.
    graph.add_edge(Verdict.REFINE_RESPONSE.value, "score_groundedness")
    graph.add_edge(Verdict.REFINE_QUERY.value, "retrieve_context")
    graph.add_edge(Verdict.EXHAUSTED.value, END)

    return graph.compile()


class AgentRunner:
    """Answers questions by driving the compiled graph."""

    def __init__(self, deps: AgentDependencies | None = None) -> None:
        """Compile once and reuse; compilation is not free."""
        self._deps = deps or build_default_dependencies()
        self._graph = build_graph(self._deps)

    @property
    def settings(self) -> Settings:
        """The settings this runner was built with."""
        return self._deps.settings

    def ask(self, question: str, *, run_id: str | None = None) -> AgentState:
        """Answer ``question`` and return the full terminal state.

        The state, not just the text, so that a caller can see the judge scores,
        the repair count and the node history. An answer whose quality gates are
        invisible is an answer you cannot act on.
        """
        state = initial_state(question, run_id=run_id or uuid.uuid4().hex[:12])
        result: AgentState = self._graph.invoke(state)
        return result
