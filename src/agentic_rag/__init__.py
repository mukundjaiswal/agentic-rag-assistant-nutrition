"""Agentic RAG assistant.

A nutrition question-answering assistant over an indexed document corpus.
The package itself is domain-neutral; the domain is the corpus.

A retrieval-augmented question-answering agent expressed as an explicit state
machine. The agent judges its own output and repairs it along one of two paths
before returning, under a bounded iteration budget.
"""

from agentic_rag.agent.state import AgentState, Verdict
from agentic_rag.eval_metrics import register_agent_metrics
from agentic_rag.settings import Settings, get_settings

__all__ = [
    "AgentState",
    "Settings",
    "Verdict",
    "__version__",
    "get_settings",
    "register_agent_metrics",
]
__version__ = "0.1.0"
