"""Internal JSON cache helpers for fetched web context.

Raw responses from TMDB/Wikipedia are cached under ``<project>/cache/`` so the
tool works fully offline once a project has been populated (AGENTS.md "Online
context"). Cache is refreshed only on explicit user request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def cache_file(cache_dir: str | os.PathLike[str], key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def read_json(path: str | os.PathLike[str]) -> Any | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
