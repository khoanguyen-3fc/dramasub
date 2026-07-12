"""Subtitle parsing and reassembly.

The LLM never sees timestamps or file structure. This module parses a
subtitle file into integer-indexed cues, exposes only the translatable text
to the rest of the pipeline, and reassembles the file programmatically —
preserving every timestamp and styling tag exactly.

Design notes
------------
* Cues are addressed by their integer position in the underlying event list.
  That index is the contract between :mod:`chunker`, :mod:`pass2` and the LLM.
* Inline ASS override tags that *wrap* a line (``{\\an8}``, ``{\\i1}...{\\i0}``)
  are split off as ``lead``/``trail`` and re-applied verbatim after
  translation, so the model only ever sees clean dialogue. Any override tags
  or ``\\N`` line breaks *inside* the body are left in place; the model is
  told to preserve them and :func:`preservation_error` validates that it did.
* Writing output is atomic and self-checked: we serialize to a temp file,
  reload it plus the source, and abort unless cue count and every timestamp
  match (see AGENTS.md "Runtime self-checks").
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pysubs2

from dramasub.core.errors import SubtitleError

logger = logging.getLogger(__name__)

# Encodings tried in order when the caller does not specify one. Korean drama
# subs in the wild are usually UTF-8 but sometimes cp949/euc-kr; latin-1 never
# raises and is the last resort so a file always loads.
_FALLBACK_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1")

_LEAD_TAGS_RE = re.compile(r"^((?:\{[^}]*\})+)")
_TRAIL_TAGS_RE = re.compile(r"((?:\{[^}]*\})+)$")
_TAG_RE = re.compile(r"\{[^}]*\}")
_LINEBREAK_TOKEN = r"\N"


def split_tags(text: str) -> tuple[str, str, str]:
    """Split *text* into ``(lead, body, trail)`` around wrapping override tags.

    ``lead``/``trail`` are the runs of ``{...}`` tags at the very start/end;
    ``body`` is everything between (and may still contain internal tags).
    """
    lead = ""
    trail = ""
    m = _LEAD_TAGS_RE.match(text)
    if m:
        lead = m.group(1)
        text = text[len(lead):]
    m = _TRAIL_TAGS_RE.search(text)
    if m:
        trail = m.group(1)
        text = text[: m.start()]
    return lead, text, trail


def internal_tags(text: str) -> list[str]:
    """Return the ``{...}`` override tags found inside *text*."""
    return _TAG_RE.findall(text)


def preservation_error(source_body: str, translated: str) -> str | None:
    """Check that a translation kept the source's inline tags and line breaks.

    Returns a short human-readable hint describing the first mismatch, or
    ``None`` if the translation is structurally faithful. Used by the retry
    loop in :mod:`pass2` to tell the model what it got wrong.
    """
    src_breaks = source_body.count(_LINEBREAK_TOKEN)
    out_breaks = translated.count(_LINEBREAK_TOKEN)
    if src_breaks != out_breaks:
        return (
            f"expected {src_breaks} '\\N' line break(s) but found {out_breaks}; "
            "preserve every '\\N' exactly"
        )
    src_tags = sorted(internal_tags(source_body))
    out_tags = sorted(internal_tags(translated))
    if src_tags != out_tags:
        return (
            f"styling tags changed: expected {src_tags or 'none'} "
            f"but found {out_tags or 'none'}; keep every '{{...}}' tag verbatim"
        )
    return None


@dataclass
class Cue:
    """A single subtitle event, addressed by its integer ``index``.

    ``body`` is the translatable dialogue with wrapping tags removed;
    :attr:`full_text` reassembles it with ``lead``/``trail``.
    """

    index: int
    start: int  # milliseconds
    end: int  # milliseconds
    style: str
    name: str
    lead: str
    body: str
    trail: str
    is_comment: bool

    @property
    def full_text(self) -> str:
        return f"{self.lead}{self.body}{self.trail}"

    @property
    def is_translatable(self) -> bool:
        return not self.is_comment and bool(self.body.strip())

    @property
    def plaintext(self) -> str:
        """Body with tags stripped and ``\\N`` rendered as a newline."""
        return _TAG_RE.sub("", self.body).replace(_LINEBREAK_TOKEN, "\n")


class SubtitleDoc:
    """A parsed subtitle file plus its integer-indexed cues."""

    def __init__(self, ssa: pysubs2.SSAFile, path: Path, fmt: str) -> None:
        self._ssa = ssa
        self.path = path
        self.format = fmt
        self.cues: list[Cue] = [_cue_from_event(i, ev) for i, ev in enumerate(ssa)]

    def __len__(self) -> int:
        return len(self.cues)

    def cue(self, index: int) -> Cue:
        try:
            return self.cues[index]
        except IndexError as exc:
            raise SubtitleError(f"no cue with index {index}") from exc

    def translatable(self) -> dict[int, str]:
        """Map of ``{index: body}`` for every cue worth translating."""
        return {c.index: c.body for c in self.cues if c.is_translatable}

    def translatable_indices(self) -> list[int]:
        return [c.index for c in self.cues if c.is_translatable]

    def apply(self, translations: dict[int, str]) -> None:
        """Set the body of each named cue and write it back to the event."""
        for index, new_body in translations.items():
            cue = self.cue(index)
            cue.body = new_body
            self._ssa[index].text = cue.full_text

    def save(self, out_path: str | os.PathLike[str], fmt: str | None = None) -> None:
        """Serialize to *out_path*.

        Format is inferred from the extension unless *fmt* is given — needed
        when writing to a temp path whose suffix is not a subtitle extension.
        """
        out_path = Path(out_path)
        self._ssa.save(
            str(out_path), encoding="utf-8", format_=fmt or _format_for(out_path)
        )


def load(
    path: str | os.PathLike[str],
    encoding: str | None = None,
    fmt: str | None = None,
) -> SubtitleDoc:
    """Parse a subtitle file into a :class:`SubtitleDoc`.

    When *encoding* is ``None`` a small set of common encodings is tried.
    Format is inferred from the extension unless *fmt* is given.
    """
    path = Path(path)
    if not path.is_file():
        raise SubtitleError(f"subtitle file not found: {path}")
    fmt = fmt or _format_for(path)
    encodings = (encoding,) if encoding else _FALLBACK_ENCODINGS
    last_error: Exception | None = None
    for enc in encodings:
        try:
            ssa = pysubs2.load(str(path), encoding=enc, format_=fmt)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        except Exception as exc:  # pysubs2 raises assorted parse errors
            raise SubtitleError(f"could not parse subtitle {path}: {exc}") from exc
        if enc != encodings[0]:
            logger.info("loaded %s using fallback encoding %s", path.name, enc)
        return SubtitleDoc(ssa, path, fmt)
    raise SubtitleError(
        f"could not decode subtitle {path} with any of {encodings}"
    ) from last_error


def verify_timing(source: SubtitleDoc, output: SubtitleDoc) -> None:
    """Assert *output* has the same cue count and timestamps as *source*.

    Raises :class:`SubtitleError` on the first discrepancy. This is the
    mandatory post-reassembly self-check.
    """
    if len(source) != len(output):
        raise SubtitleError(
            f"cue count changed during reassembly: source has {len(source)} "
            f"cues, output has {len(output)}"
        )
    for src, out in zip(source.cues, output.cues):
        if src.start != out.start or src.end != out.end:
            raise SubtitleError(
                f"timestamp mismatch at cue {src.index}: source "
                f"{src.start}-{src.end}ms, output {out.start}-{out.end}ms"
            )


def save_verified(
    doc: SubtitleDoc,
    out_path: str | os.PathLike[str],
    source: SubtitleDoc,
) -> None:
    """Atomically write *doc*, verifying timing against *source* first.

    Serializes to a temp file, reloads it, and only moves it into place once
    the reloaded output matches the source's cue count and every timestamp.
    On mismatch the temp file is removed and :class:`SubtitleError` is raised —
    a bad output file is never left behind.
    """
    out_path = Path(out_path)
    fmt = _format_for(out_path)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    doc.save(tmp_path, fmt=fmt)
    try:
        reloaded = load(tmp_path, encoding="utf-8", fmt=fmt)
        verify_timing(source, reloaded)
    except SubtitleError:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, out_path)
    logger.info("wrote %d cues to %s (timing verified)", len(doc), out_path)


def _cue_from_event(index: int, ev: pysubs2.SSAEvent) -> Cue:
    lead, body, trail = split_tags(ev.text)
    return Cue(
        index=index,
        start=ev.start,
        end=ev.end,
        style=ev.style,
        name=ev.name,
        lead=lead,
        body=body,
        trail=trail,
        is_comment=ev.is_comment,
    )


def _format_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("srt",):
        return "srt"
    if suffix in ("ass", "ssa"):
        return "ass"
    if suffix in ("vtt",):
        return "vtt"
    raise SubtitleError(
        f"unsupported subtitle extension '{path.suffix}' (expected .srt/.ass/.ssa)"
    )
