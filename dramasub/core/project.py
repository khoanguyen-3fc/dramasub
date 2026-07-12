"""Project (show) lifecycle: create, load, save.

Every show is a project — a directory that accumulates context across
episodes. Languages are configuration, not assumptions: ``source_language``
and ``target_language`` are read from ``project.yaml`` (defaults ``ko`` →
``vi``) and threaded through every prompt. Nothing here is language-specific.

Directory layout (see AGENTS.md)::

    <project>/
      project.yaml
      bible.yaml
      cache/
      episodes/e01/{source.srt,context.yaml,output.srt,summary.txt}
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from dramasub.core._yaml import read_yaml, write_yaml
from dramasub.core.bible import Bible, load_bible, new_bible, save_bible
from dramasub.core.errors import ProjectError, ValidationError

logger = logging.getLogger(__name__)

HONORIFIC_POLICIES = ("translate", "keep_romanized")

# Config defaults. Model name lives here and nowhere else — never hardcode it
# in pipeline logic (AGENTS.md "LLM usage").
DEFAULT_CONFIG: dict[str, Any] = {
    "title": "",
    "tmdb_id": None,
    "source_language": "ko",
    "target_language": "vi",
    "model": "qwen3.6:latest",
    "num_ctx": 16384,
    "keep_alive": "30m",
    "ollama_host": "http://localhost:11434",
    "honorific_policy": "translate",
    "temperature": {"pass1": 0.7, "pass2": 0.3, "summary": 0.7},
    "chunk": {"size": 12, "prev_window": 8, "lookahead": 4},
    "max_line_chars": 42,
}

PROJECT_FILE = "project.yaml"
BIBLE_FILE = "bible.yaml"
CACHE_DIR = "cache"
EPISODES_DIR = "episodes"


class Project:
    """A loaded show project. Pass this object explicitly; never global."""

    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        self.config = config

    # -- config accessors (fall back to defaults for missing keys) ---------
    @property
    def title(self) -> str:
        return self.config.get("title", "")

    @property
    def tmdb_id(self) -> int | None:
        return self.config.get("tmdb_id")

    @property
    def source_language(self) -> str:
        return self.config.get("source_language", DEFAULT_CONFIG["source_language"])

    @property
    def target_language(self) -> str:
        return self.config.get("target_language", DEFAULT_CONFIG["target_language"])

    @property
    def model(self) -> str:
        return self.config.get("model", DEFAULT_CONFIG["model"])

    @property
    def num_ctx(self) -> int:
        return int(self.config.get("num_ctx", DEFAULT_CONFIG["num_ctx"]))

    @property
    def keep_alive(self) -> str:
        return self.config.get("keep_alive", DEFAULT_CONFIG["keep_alive"])

    @property
    def ollama_host(self) -> str:
        return self.config.get("ollama_host", DEFAULT_CONFIG["ollama_host"])

    @property
    def honorific_policy(self) -> str:
        return self.config.get("honorific_policy", DEFAULT_CONFIG["honorific_policy"])

    @property
    def max_line_chars(self) -> int:
        return int(self.config.get("max_line_chars", DEFAULT_CONFIG["max_line_chars"]))

    def temperature(self, pass_name: str) -> float:
        temps = self.config.get("temperature", {}) or {}
        default = DEFAULT_CONFIG["temperature"][pass_name]
        return float(temps.get(pass_name, default))

    def chunk_setting(self, key: str) -> int:
        chunk = self.config.get("chunk", {}) or {}
        return int(chunk.get(key, DEFAULT_CONFIG["chunk"][key]))

    # -- paths -------------------------------------------------------------
    @property
    def project_yaml(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def bible_yaml(self) -> Path:
        return self.root / BIBLE_FILE

    @property
    def cache_dir(self) -> Path:
        return self.root / CACHE_DIR

    @property
    def episodes_dir(self) -> Path:
        return self.root / EPISODES_DIR

    def episode_dir(self, number: int) -> Path:
        return self.episodes_dir / _episode_name(number)

    def episode_source(self, number: int) -> Path:
        return self.episode_dir(number) / "source.srt"

    def episode_context(self, number: int) -> Path:
        return self.episode_dir(number) / "context.yaml"

    def episode_output(self, number: int) -> Path:
        return self.episode_dir(number) / "output.srt"

    def episode_summary(self, number: int) -> Path:
        return self.episode_dir(number) / "summary.txt"

    def episode_source_existing(self, number: int) -> Path | None:
        """The imported source file for an episode, whatever its extension."""
        return _first_existing(self.episode_dir(number), "source")

    def episode_output_existing(self, number: int) -> Path | None:
        """The written output file for an episode, whatever its extension."""
        return _first_existing(self.episode_dir(number), "output")

    def list_episodes(self) -> list[int]:
        """Episode numbers with a directory on disk, ascending."""
        if not self.episodes_dir.is_dir():
            return []
        numbers = []
        for child in self.episodes_dir.iterdir():
            n = _parse_episode_name(child.name)
            if n is not None and child.is_dir():
                numbers.append(n)
        return sorted(numbers)

    # -- bible -------------------------------------------------------------
    def load_bible(self) -> Bible:
        if self.bible_yaml.is_file():
            return load_bible(self.bible_yaml)
        return new_bible()

    def save_bible(self, bible: Bible) -> None:
        save_bible(bible, self.bible_yaml)

    # -- episode setup -----------------------------------------------------
    def import_episode_source(self, number: int, src_file: str | Path) -> Path:
        """Copy *src_file* into ``episodes/eNN/source.srt`` (never modified after)."""
        src_file = Path(src_file)
        if not src_file.is_file():
            raise ProjectError(f"subtitle file not found: {src_file}")
        dest = self.episode_source(number)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the source verbatim; keep its extension if not .srt.
        if src_file.suffix.lower() != ".srt":
            dest = dest.with_suffix(src_file.suffix.lower())
        shutil.copy2(src_file, dest)
        logger.info("imported episode %d source: %s -> %s", number, src_file, dest)
        return dest

    def save(self) -> None:
        write_yaml(self.project_yaml, self.config)


def create(
    root: str | Path,
    *,
    title: str,
    tmdb_id: int | None = None,
    source_language: str = "ko",
    target_language: str = "vi",
    model: str | None = None,
) -> Project:
    """Create a new project directory. Fails if one already exists."""
    root = Path(root)
    if (root / PROJECT_FILE).exists():
        raise ProjectError(f"project already exists at {root}")

    config = dict(DEFAULT_CONFIG)
    config["temperature"] = dict(DEFAULT_CONFIG["temperature"])
    config["chunk"] = dict(DEFAULT_CONFIG["chunk"])
    config["title"] = title
    config["tmdb_id"] = tmdb_id
    config["source_language"] = source_language
    config["target_language"] = target_language
    if model:
        config["model"] = model
    validate_config(config)

    root.mkdir(parents=True, exist_ok=True)
    (root / CACHE_DIR).mkdir(exist_ok=True)
    (root / EPISODES_DIR).mkdir(exist_ok=True)

    project = Project(root, config)
    project.save()
    project.save_bible(new_bible())
    logger.info("created project %r at %s (%s->%s)", title, root, source_language, target_language)
    return project


def load(root: str | Path) -> Project:
    """Load an existing project."""
    root = Path(root)
    project_file = root / PROJECT_FILE
    if not project_file.is_file():
        raise ProjectError(f"no project at {root} (missing {PROJECT_FILE})")
    config = read_yaml(project_file)
    validate_config(config)
    return Project(root, config)


def validate_config(config: dict[str, Any]) -> None:
    """Explicit validation — no pydantic. Raises :class:`ValidationError`."""
    for key in ("source_language", "target_language", "model"):
        value = config.get(key)
        if not value or not isinstance(value, str):
            raise ValidationError(f"project config: {key!r} must be a non-empty string")
    policy = config.get("honorific_policy", DEFAULT_CONFIG["honorific_policy"])
    if policy not in HONORIFIC_POLICIES:
        raise ValidationError(
            f"project config: honorific_policy must be one of {HONORIFIC_POLICIES}, "
            f"got {policy!r}"
        )
    num_ctx = config.get("num_ctx", DEFAULT_CONFIG["num_ctx"])
    if not isinstance(num_ctx, int) or num_ctx <= 0:
        raise ValidationError(f"project config: num_ctx must be a positive int, got {num_ctx!r}")
    tmdb_id = config.get("tmdb_id")
    if tmdb_id is not None and not isinstance(tmdb_id, int):
        raise ValidationError(f"project config: tmdb_id must be an int or null, got {tmdb_id!r}")


def _first_existing(directory: Path, stem: str) -> Path | None:
    for suffix in (".srt", ".ass", ".ssa", ".vtt"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _episode_name(number: int) -> str:
    if number < 1:
        raise ProjectError(f"episode number must be >= 1, got {number}")
    return f"e{number:02d}"


def _parse_episode_name(name: str) -> int | None:
    if len(name) >= 2 and name[0] == "e" and name[1:].isdigit():
        return int(name[1:])
    return None
