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
import os
import shutil
from pathlib import Path
from typing import Any

from dramasub.core._yaml import read_yaml, write_yaml
from dramasub.core.bible import Bible, load_bible, new_bible, save_bible
from dramasub.core.errors import ProjectError, ValidationError
from dramasub.core.lang import language_name

logger = logging.getLogger(__name__)

HONORIFIC_POLICIES = ("translate", "keep_romanized")
LOANWORD_POLICIES = ("keep_english", "localize")
ROMANIZATION_STYLES = ("media", "revised")

# Config defaults. Model name lives here and nowhere else — never hardcode it
# in pipeline logic (AGENTS.md "LLM usage").
DEFAULT_CONFIG: dict[str, Any] = {
    "title": "",
    "tmdb_id": None,
    "source_language": "ko",
    "target_language": "vi",
    "model": "gemma4:latest",
    "num_ctx": 16384,
    "keep_alive": "30m",
    "ollama_host": "http://localhost:11434",
    "honorific_policy": "translate",
    "loanword_policy": "keep_english",
    "romanization": "media",
    # Translation guide + base dictionary injected into prompts. "default"
    # resolves to the packaged file for the project's language pair when one
    # ships (e.g. ko->vi K-drama); "none" disables; any other value is a path
    # (relative to the project root) to a user-supplied file — the way to
    # support other genres or language pairs.
    "guide": "default",
    "dictionary": "default",
    "temperature": {"pass1": 0.7, "pass2": 0.3, "summary": 0.7},
    # Temperatures for pass-2 retry attempts (first retry, second retry, ...).
    # Hotter retries help escape a stubborn wrong output the model keeps
    # regenerating at low temperature; the first attempt uses temperature.pass2.
    "retry_temperatures": [0.55, 0.8],
    "chunk": {"size": 12, "prev_window": 8, "lookahead": 4},
    # Max characters PER LINE — the standard professional-subtitling convention
    # (~42 chars/line for Latin-script languages, e.g. Netflix's guideline); with
    # the 2-line cap that is an ~84-char budget per cue. This single value is the
    # one source of truth: rewrap/_balance (subtitle.py), the pass-2 tighten pass,
    # and QC's check_length all take it from here — no literal 42 lives in the
    # algorithms, so overriding `max_line_chars` in project.yaml changes them all.
    # Measured as Unicode codepoints of the tag-stripped text (fine for NFC
    # Vietnamese); it is a line-LENGTH limit, not a reading-speed (CPS) limit.
    "max_line_chars": 42,
}

PROJECT_FILE = "project.yaml"
BIBLE_FILE = "bible.yaml"
CACHE_DIR = "cache"
EPISODES_DIR = "episodes"

# Packaged default guides/dictionaries, keyed by file name per language pair.
DEFAULTS_DIR = Path(__file__).parent / "defaults"


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
        """Ollama server URL. ``OLLAMA_HOST`` (env / .env) overrides the config."""
        host = os.environ.get("OLLAMA_HOST") or self.config.get(
            "ollama_host", DEFAULT_CONFIG["ollama_host"]
        )
        host = host.strip().rstrip("/")
        if "://" not in host:  # accept bare host:port from .env
            host = "http://" + host
        return host

    @property
    def honorific_policy(self) -> str:
        return self.config.get("honorific_policy", DEFAULT_CONFIG["honorific_policy"])

    @property
    def loanword_policy(self) -> str:
        return self.config.get("loanword_policy", DEFAULT_CONFIG["loanword_policy"])

    @property
    def romanization(self) -> str:
        return self.config.get("romanization", DEFAULT_CONFIG["romanization"])

    def style_guidance(self) -> str:
        """Prompt-ready guidance for loanwords and name romanization."""
        source = language_name(self.source_language)
        target = language_name(self.target_language)
        parts = []
        if self.loanword_policy == "keep_english":
            parts.append(
                f"Keep loanwords that {source} speakers themselves "
                "use in English (e.g. TF, KPI, highball, prototype, retro) rather "
                f"than forcing a {target} word."
            )
        else:
            parts.append(
                f"Prefer natural {target} words over English "
                "loanwords wherever it does not sound stiff."
            )
        if self.romanization == "media":
            parts.append(
                "Romanize personal names using common media/passport spellings "
                "(e.g. Hong Gil Dong), not academic romanization."
            )
        else:
            parts.append(
                "Romanize personal names using the standard academic "
                f"romanization for {source}."
            )
        return " ".join(parts)

    @property
    def max_line_chars(self) -> int:
        return int(self.config.get("max_line_chars", DEFAULT_CONFIG["max_line_chars"]))

    def temperature(self, pass_name: str) -> float:
        temps = self.config.get("temperature", {}) or {}
        default = DEFAULT_CONFIG["temperature"][pass_name]
        return float(temps.get(pass_name, default))

    @property
    def retry_temperatures(self) -> list[float]:
        """Per-retry temperatures for pass 2 (may be empty: retries stay cool)."""
        values = self.config.get(
            "retry_temperatures", DEFAULT_CONFIG["retry_temperatures"]
        )
        return [float(v) for v in values]

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

    def load_episode_context(self, number: int) -> dict[str, Any] | None:
        """The saved pass-1 context for an episode, or ``None`` if never run."""
        path = self.episode_context(number)
        if not path.is_file():
            return None
        return read_yaml(path)

    # -- guide / dictionary --------------------------------------------------
    def guide_path(self) -> Path | None:
        """Resolved translation-guide file, or ``None`` when disabled/absent."""
        return self._resolve_asset(
            "guide", f"guide.{self.source_language}-{self.target_language}.txt"
        )

    def dictionary_path(self) -> Path | None:
        """Resolved base-dictionary file, or ``None`` when disabled/absent."""
        return self._resolve_asset(
            "dictionary",
            f"dictionary.{self.source_language}-{self.target_language}.yaml",
        )

    def load_guide(self) -> str:
        """Translation-guide text for prompts ('' when none configured)."""
        path = self.guide_path()
        if path is None:
            return ""
        lines = path.read_text(encoding="utf-8").splitlines()
        body = [ln for ln in lines if not ln.startswith("#")]
        return "\n".join(body).strip()

    def load_dictionary(self) -> list[dict[str, Any]]:
        """Base dictionary entries (``{source, target, note?}``); may be empty.

        These guide generation like glossary entries but are NOT enforced by
        QC — only the hand-curated project bible glossary is.
        """
        path = self.dictionary_path()
        if path is None:
            return []
        terms = read_yaml(path).get("terms", [])
        return [
            t for t in terms
            if isinstance(t, dict) and t.get("source") and t.get("target")
        ]

    def _resolve_asset(self, key: str, packaged_name: str) -> Path | None:
        value = self.config.get(key, DEFAULT_CONFIG[key])
        if value in (None, "", "none"):
            return None
        if value == "default":
            packaged = DEFAULTS_DIR / packaged_name
            # No packaged file for this language pair -> silently none; users
            # supply their own via a path for other pairs/genres.
            return packaged if packaged.is_file() else None
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise ProjectError(f"{key} file not found: {path}")
        return path

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
    for key, allowed in (
        ("loanword_policy", LOANWORD_POLICIES),
        ("romanization", ROMANIZATION_STYLES),
    ):
        value = config.get(key, DEFAULT_CONFIG[key])
        if value not in allowed:
            raise ValidationError(
                f"project config: {key} must be one of {allowed}, got {value!r}"
            )
    num_ctx = config.get("num_ctx", DEFAULT_CONFIG["num_ctx"])
    if not isinstance(num_ctx, int) or num_ctx <= 0:
        raise ValidationError(f"project config: num_ctx must be a positive int, got {num_ctx!r}")
    for key in ("guide", "dictionary"):
        value = config.get(key, DEFAULT_CONFIG[key])
        if value is not None and not isinstance(value, str):
            raise ValidationError(
                f"project config: {key} must be 'default', 'none', or a file "
                f"path, got {value!r}"
            )
    retry_temps = config.get("retry_temperatures", DEFAULT_CONFIG["retry_temperatures"])
    if not isinstance(retry_temps, list) or not all(
        isinstance(v, (int, float)) and 0 <= v <= 2 for v in retry_temps
    ):
        raise ValidationError(
            "project config: retry_temperatures must be a list of numbers "
            f"between 0 and 2, got {retry_temps!r}"
        )
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
