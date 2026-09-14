"""Deterministic chat model for tests.

Exists so that the whole agent can be exercised without a network call, an API
key, or a paid token. A test that cannot run in CI is a test that stops being
run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


class ScriptedChatModel:
    """Returns pre-set replies in order, then repeats the last one.

    Also records every prompt it was given, so a test can assert on what the
    agent actually asked rather than only on what came back.
    """

    def __init__(
        self,
        replies: Iterable[str] | None = None,
        *,
        responder: Callable[[str], str] | None = None,
    ) -> None:
        """Configure either a fixed reply sequence or a prompt-driven responder."""
        self._replies = list(replies or [])
        self._responder = responder
        self._index = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        """Record ``prompt`` and return the next scripted reply."""
        del temperature
        self.prompts.append(prompt)
        if self._responder is not None:
            return self._responder(prompt)
        if not self._replies:
            return ""
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return reply

    @property
    def call_count(self) -> int:
        """Number of completions requested so far."""
        return len(self.prompts)
