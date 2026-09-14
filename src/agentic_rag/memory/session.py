"""Session-scoped conversational memory.

Bounded on purpose. An unbounded transcript grows the prompt on every turn until
it crowds out the retrieved context, which shows up as answers getting worse the
longer a conversation runs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Turn:
    """One question and the answer given."""

    question: str
    response: str


@dataclass
class SessionMemory:
    """A bounded window of recent turns for one session."""

    session_id: str
    max_turns: int = 10
    _turns: deque[Turn] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        """Enforce the window size."""
        if self.max_turns < 1:
            message = "max_turns must be at least 1"
            raise ValueError(message)
        self._turns = deque(self._turns, maxlen=self.max_turns)

    @property
    def turns(self) -> list[Turn]:
        """Recent turns, oldest first."""
        return list(self._turns)

    def add(self, question: str, response: str) -> None:
        """Record a turn, evicting the oldest once the window is full."""
        self._turns.append(Turn(question=question, response=response))

    def as_context(self) -> str:
        """Render the window as prompt-ready text."""
        return "\n".join(f"Q: {t.question}\nA: {t.response}" for t in self._turns)

    def clear(self) -> None:
        """Forget every turn."""
        self._turns.clear()

    def __len__(self) -> int:
        """Number of turns currently held."""
        return len(self._turns)
