"""Streamlit chat interface.

Surfaces the judge scores and the repair count next to every answer. If the
quality gates are invisible to the person reading the answer, the harness may as
well not be there.
"""

from __future__ import annotations

import streamlit as st

from agentic_rag.agent.graph import AgentRunner
from agentic_rag.logging_config import configure_logging
from agentic_rag.memory.session import SessionMemory
from agentic_rag.safety.guardrails import check_input
from agentic_rag.settings import get_settings


def _bootstrap() -> tuple[AgentRunner, SessionMemory]:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    if "runner" not in st.session_state:
        st.session_state.runner = AgentRunner()
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory(session_id="streamlit")
    return st.session_state.runner, st.session_state.memory


def render() -> None:
    """Draw the chat interface."""
    st.set_page_config(page_title="Agentic RAG Assistant", layout="centered")
    st.title("Agentic RAG Assistant")

    runner, memory = _bootstrap()

    with st.sidebar:
        st.caption("Configuration")
        st.json(
            {
                "model": runner.settings.llm_model,
                "groundedness_threshold": runner.settings.groundedness_threshold,
                "precision_threshold": runner.settings.precision_threshold,
                "max_iterations": runner.settings.max_iterations,
                "safety_filter": "on"
                if runner.settings.enable_safety_filter
                else "off",
            }
        )
        if st.button("Clear conversation"):
            memory.clear()
            st.rerun()

    for turn in memory.turns:
        st.chat_message("user").write(turn.question)
        st.chat_message("assistant").write(turn.response)

    question = st.chat_input("Ask a question about the indexed documents")
    if not question:
        return

    verdict = check_input(question, settings=runner.settings)
    if not verdict.allowed:
        st.chat_message("assistant").write(
            f"This question was not processed: {verdict.reason}"
        )
        return

    st.chat_message("user").write(question)
    with st.spinner("Retrieving and checking..."):
        state = runner.ask(question)

    response = state.get("response", "")
    st.chat_message("assistant").write(response)

    with st.expander("Quality gates"):
        st.json(
            {
                "run_id": state.get("run_id"),
                "groundedness": state.get("groundedness"),
                "precision": state.get("precision"),
                "repairs": state.get("iteration", 0),
                "stopped_at_budget": state.get("exhausted", False),
                "screened": verdict.was_screened,
                "path": state.get("history", []),
            }
        )

    memory.add(question, response)


render()
