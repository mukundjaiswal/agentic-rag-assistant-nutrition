"""Chat-model adapters."""

from agentic_rag.llm.fake import ScriptedChatModel
from agentic_rag.llm.openai_client import OpenAIChatModel

__all__ = ["OpenAIChatModel", "ScriptedChatModel"]
