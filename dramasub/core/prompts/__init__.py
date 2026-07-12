"""Prompt template loading.

Every prompt lives in its own ``.txt`` file in this directory with a version
header, so prompt content is never scattered as inline strings through the
pipeline logic (AGENTS.md "Coding conventions"). When a prompt changes, bump
the version note in that file.

File format::

    # <name> prompt
    # version: N — what changed
    ---
    <template body, using ${placeholder} substitution>

Everything up to the first line that is exactly ``---`` is a header comment and
is stripped at load time. The body uses :class:`string.Template` ``${name}``
placeholders — deliberately *not* ``str.format`` ``{}``, which would collide
with the ``{...}`` ASS override tags and JSON braces that appear in prompts.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from dramasub.core.errors import DramasubError

_PROMPT_DIR = Path(__file__).parent
_HEADER_SEP = "---"


class PromptError(DramasubError):
    """A prompt template is missing or references an undefined placeholder."""


def load_prompt(name: str) -> str:
    """Return the template body of ``<name>.txt`` (header stripped)."""
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        raise PromptError(f"prompt template not found: {path}")
    return _strip_header(path.read_text(encoding="utf-8"))


def render(name: str, /, **values: Any) -> str:
    """Load prompt *name* and substitute ``${...}`` placeholders.

    Uses strict substitution so a missing variable is a loud error rather than
    a silently malformed prompt sent to the model.
    """
    template = Template(load_prompt(name))
    try:
        return template.substitute(_stringify(values))
    except KeyError as exc:
        raise PromptError(f"prompt {name!r} is missing value for {exc}") from exc
    except ValueError as exc:
        raise PromptError(f"prompt {name!r} has a malformed placeholder: {exc}") from exc


def _strip_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() == _HEADER_SEP:
            return "".join(lines[i + 1:])
    # No separator: treat the whole file as body.
    return text


def _stringify(values: dict[str, Any]) -> dict[str, str]:
    return {k: ("" if v is None else str(v)) for k, v in values.items()}
