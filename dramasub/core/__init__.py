"""Pure library for dramasub.

Rules enforced by convention (see AGENTS.md):

* No UI imports, no ``print()``. Functions raise or return.
* Errors are :class:`~dramasub.core.errors.DramasubError` subclasses so
  ``cli.py`` can catch and present them.
* Logging goes through the stdlib :mod:`logging` module.
"""

from dramasub.core.errors import (
    ChunkValidationError,
    ContextFetchError,
    DramasubError,
    LLMError,
    ProjectError,
    SubtitleError,
    ValidationError,
)

__all__ = [
    "DramasubError",
    "ProjectError",
    "SubtitleError",
    "LLMError",
    "ChunkValidationError",
    "ValidationError",
    "ContextFetchError",
]
