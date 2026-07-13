"""Language-code helpers.

Config stores short codes (``ko``, ``vi``) — languages are configuration, not
assumptions — but prompts read better with full English names: a model follows
"translate into Vietnamese" more reliably than "translate into vi". Everything
that renders a prompt or a user-facing message should go through
:func:`language_name`; everything that branches on the language (script
detection, API calls) keeps using the code.
"""

from __future__ import annotations

_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "ms": "Malay",
    "my": "Burmese",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tl": "Filipino",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def language_name(code: str) -> str:
    """English name for a language code, falling back to the code itself."""
    return _NAMES.get(code.split("-")[0].strip().lower(), code)
