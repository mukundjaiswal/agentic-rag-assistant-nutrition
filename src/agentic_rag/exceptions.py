"""Exception hierarchy.

A single base class lets a caller distinguish this package's failures from
arbitrary runtime errors without catching ``Exception``.
"""


class AgenticRagError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(AgenticRagError):
    """Required configuration is missing or invalid."""


class IngestionError(AgenticRagError):
    """A document could not be parsed, chunked or indexed."""


class RetrievalError(AgenticRagError):
    """The vector store could not be reached or returned nothing usable."""


class JudgeParseError(AgenticRagError):
    """A judge returned something that is not a score in ``[0, 1]``.

    Raised only where a caller has asked for strict parsing. The graph itself
    prefers the non-raising path: an unreadable verdict routes to repair rather
    than aborting the run.
    """


class SafetyFilterError(AgenticRagError):
    """The safety filter is enabled but not usable."""
