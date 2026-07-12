"""Quality-control checks over translated output.

These are runtime self-checks, not a test suite (AGENTS.md forbids one). They
run at the end of every ``translate`` and are also exposed standalone via the
``qc`` CLI command. All checks compare the output against the source by cue
index, so timing/structure integrity is assumed already verified by
:func:`dramasub.core.subtitle.save_verified`.

Checks:

* **untranslated** — an output cue whose body is byte-identical to the source
  (a chunk that failed and was skipped, or the model echoed the input).
* **empty** — a source cue that has dialogue but whose output body is blank.
* **length** — a rendered output line longer than the soft limit (~42 chars).
  Warned, never truncated.
* **glossary** — the source term appears in a cue but its agreed
  target-language rendering is absent from the corresponding output cue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dramasub.core.bible import Bible
from dramasub.core.subtitle import SubtitleDoc

logger = logging.getLogger(__name__)


@dataclass
class QCWarning:
    kind: str  # "untranslated" | "empty" | "length" | "glossary"
    detail: str
    index: int | None = None

    def __str__(self) -> str:
        where = f"cue {self.index}: " if self.index is not None else ""
        return f"[{self.kind}] {where}{self.detail}"


def run_qc(
    bible: Bible,
    source: SubtitleDoc,
    output: SubtitleDoc,
    *,
    max_line_chars: int,
) -> list[QCWarning]:
    """Run every check and return the collected warnings, ordered by cue."""
    warnings: list[QCWarning] = []
    warnings += check_coverage(source, output)
    warnings += check_length(output, max_line_chars)
    warnings += check_glossary(bible, source, output)
    warnings.sort(key=lambda w: (w.index if w.index is not None else -1, w.kind))
    logger.info("QC produced %d warning(s)", len(warnings))
    return warnings


def check_coverage(source: SubtitleDoc, output: SubtitleDoc) -> list[QCWarning]:
    """Flag translatable cues that came back empty or untranslated."""
    warnings: list[QCWarning] = []
    for src in source.cues:
        if not src.is_translatable:
            continue
        out = output.cue(src.index)
        if not out.body.strip():
            warnings.append(QCWarning("empty", "output cue is blank", src.index))
        elif out.body.strip() == src.body.strip():
            warnings.append(
                QCWarning("untranslated", "output is identical to the source", src.index)
            )
    return warnings


def check_length(output: SubtitleDoc, max_line_chars: int) -> list[QCWarning]:
    """Flag rendered output lines over the soft per-line limit."""
    warnings: list[QCWarning] = []
    for cue in output.cues:
        if not cue.body.strip():
            continue
        for line in cue.plaintext.split("\n"):
            n = len(line)
            if n > max_line_chars:
                warnings.append(
                    QCWarning(
                        "length",
                        f"rendered line is {n} chars (soft limit {max_line_chars}): "
                        f"{line!r}",
                        cue.index,
                    )
                )
    return warnings


def check_glossary(
    bible: Bible, source: SubtitleDoc, output: SubtitleDoc
) -> list[QCWarning]:
    """Flag cues where a glossary source term wasn't rendered as agreed.

    For each glossary entry, if the source term appears in a source cue, the
    matching output cue should contain the target rendering.
    """
    entries = [
        (e["source"], e["target"])
        for e in bible.glossary
        if e.get("source") and e.get("target")
    ]
    if not entries:
        return []

    warnings: list[QCWarning] = []
    for src in source.cues:
        if not src.is_translatable:
            continue
        src_text = src.plaintext
        out_text = output.cue(src.index).plaintext.lower()
        for term, rendering in entries:
            if term in src_text and rendering.lower() not in out_text:
                warnings.append(
                    QCWarning(
                        "glossary",
                        f"source term {term!r} present but rendering "
                        f"{rendering!r} missing from output",
                        src.index,
                    )
                )
    return warnings
