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
* **foreign** — an output cue contains characters from a script the target
  language doesn't use (e.g. a Chinese idiom leaking into Vietnamese — a known
  habit of Chinese-trained models under idiomatic pressure).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from dramasub.core.bible import Bible
from dramasub.core.lang import language_name
from dramasub.core.subtitle import SubtitleDoc

logger = logging.getLogger(__name__)

# Script blocks that signal a language leak when the target doesn't use them.
_SCRIPTS = {
    "Chinese": re.compile(r"[㐀-䶿一-鿿豈-﫿]"),
    "Korean": re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯]"),
    "Japanese kana": re.compile(r"[぀-ヿ]"),
}
# Scripts native to a target language (matched on the primary subtag), so a
# zh/ja/ko target is never flagged for its own writing system.
_NATIVE_SCRIPTS = {
    "zh": {"Chinese"},
    "ja": {"Chinese", "Japanese kana"},
    "ko": {"Korean", "Chinese"},
}


@dataclass
class QCWarning:
    kind: str  # "untranslated" | "empty" | "length" | "glossary" | "foreign"
    detail: str
    index: int | None = None

    def __str__(self) -> str:
        where = f"cue {self.index}: " if self.index is not None else ""
        return f"[{self.kind}] {where}{self.detail}"


def foreign_script(text: str, target_language: str) -> str:
    """Characters in *text* from scripts *target_language* doesn't use.

    Returns the offending characters (empty string if none). Latin-script
    additions such as English loanwords are deliberately not flagged — the
    loanword policy governs those; this catches script-level leaks only.
    """
    native = _NATIVE_SCRIPTS.get(target_language.split("-")[0].lower(), set())
    found: list[str] = []
    for name, rx in _SCRIPTS.items():
        if name not in native:
            found.extend(rx.findall(text))
    return "".join(found)


def run_qc(
    bible: Bible,
    source: SubtitleDoc,
    output: SubtitleDoc,
    *,
    max_line_chars: int,
    target_language: str,
) -> list[QCWarning]:
    """Run every check and return the collected warnings, ordered by cue."""
    warnings: list[QCWarning] = []
    warnings += check_coverage(source, output)
    warnings += check_length(output, max_line_chars)
    warnings += check_glossary(bible, source, output)
    warnings += check_foreign_script(source, output, target_language)
    warnings.sort(key=lambda w: (w.index if w.index is not None else -1, w.kind))
    logger.info("QC produced %d warning(s)", len(warnings))
    return warnings


def check_foreign_script(
    source: SubtitleDoc, output: SubtitleDoc, target_language: str
) -> list[QCWarning]:
    """Flag output cues containing a script the target language doesn't use.

    Cues left identical to the source (failed/skipped chunks) are skipped —
    those are already reported as ``untranslated``.
    """
    warnings: list[QCWarning] = []
    for cue in output.cues:
        if not cue.body.strip():
            continue
        if cue.body.strip() == source.cue(cue.index).body.strip():
            continue
        leak = foreign_script(cue.plaintext, target_language)
        if leak:
            warnings.append(
                QCWarning(
                    "foreign",
                    f"contains non-{language_name(target_language)} script {leak!r}",
                    cue.index,
                )
            )
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
    """Flag over-long rendered lines and cues rendered as more than two lines."""
    warnings: list[QCWarning] = []
    for cue in output.cues:
        if not cue.body.strip():
            continue
        lines = cue.plaintext.split("\n")
        if len(lines) > 2:
            warnings.append(
                QCWarning(
                    "length",
                    f"cue rendered as {len(lines)} lines (subtitles should be "
                    "at most 2)",
                    cue.index,
                )
            )
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
