"""Typed exceptions for dramasub.

``core`` raises these; :mod:`dramasub.cli` catches :class:`DramasubError`
and turns it into a friendly message + non-zero exit code. Never let a bare
exception escape ``core`` when a typed one carries more intent.
"""

from __future__ import annotations


class DramasubError(Exception):
    """Base class for every error raised by :mod:`dramasub.core`."""


class ProjectError(DramasubError):
    """A project directory is missing, malformed, or misconfigured."""


class SubtitleError(DramasubError):
    """A subtitle file could not be parsed, or failed an integrity check."""


class LLMError(DramasubError):
    """The LLM backend was unreachable or returned an unusable response."""


class ChunkValidationError(DramasubError):
    """A translated chunk failed validation after all retries."""


class ValidationError(DramasubError):
    """A config or data structure failed explicit validation."""


class ContextFetchError(DramasubError):
    """An online context source (TMDB, Wikipedia) could not be reached."""
