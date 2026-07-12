"""Internal YAML I/O helpers.

All user-editable files (``project.yaml``, ``bible.yaml``, per-episode
``context.yaml``) round-trip through here. We use PyYAML's safe loader/dumper
with:

* ``allow_unicode=True`` — keep Korean/Vietnamese text readable, not escaped.
* ``sort_keys=False`` — preserve insertion order so hand-edited files and
  unknown keys survive a round-trip in place.
* a wide line width — don't hard-wrap long notes.

Writes are atomic (temp file + ``os.replace``) so a crash never leaves a
half-written config behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from dramasub.core.errors import ValidationError


def read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping, returning ``{}`` for an empty file."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValidationError(f"expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def write_yaml(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    """Atomically dump *data* as human-editable YAML."""
    path = Path(path)
    text = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
