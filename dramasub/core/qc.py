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
* **gender** — a pronoun problem: an output cue that hedges with a slashed
  address pair (``anh/cô``) instead of committing to one, or a bible address
  row whose gendered term contradicts the known gender of that character.
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

# Vietnamese relational address terms that carry a fixed gender, used by the
# gender check. Neutral terms (tôi, em, tớ, cậu, con, mình, cháu, bác) are
# deliberately excluded — they contradict no gender.
_MALE_TERMS = {"anh", "ông", "chú"}
_FEMALE_TERMS = {"cô", "chị", "bà", "dì"}
# Every relational pronoun, for spotting an unresolved "anh/cô"-style hedge in
# output: a subtitle must commit to one term, so a slashed pronoun pair is
# always an error (longest-first so the alternation is greedy).
_PRONOUNS = _MALE_TERMS | _FEMALE_TERMS | {
    "em", "tôi", "tớ", "cậu", "con", "mình", "cháu", "bác",
}
_PRON_ALT = "|".join(sorted(_PRONOUNS, key=len, reverse=True))
_HEDGE_RE = re.compile(
    rf"\b(?:{_PRON_ALT})\s*/\s*(?:{_PRON_ALT})\b", re.IGNORECASE
)


@dataclass
class QCWarning:
    kind: str  # untranslated | empty | length | glossary | foreign | gender
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
    warnings += check_gender_consistency(bible, output)
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


def check_gender_consistency(bible: Bible, output: SubtitleDoc) -> list[QCWarning]:
    """Flag pronoun-gender problems.

    Two independent checks:

    * **output hedges** — a cue that leaves the pronoun unresolved as a slashed
      pair (``anh/cô``, ``em/tôi``). Subtitles must commit to one term; a
      slashed pair is always an error, whichever side is right.
    * **address-table contradictions** — a bible address row whose gendered
      ``self``/``other`` term (e.g. ``anh``) disagrees with the known gender of
      the character it applies to. These rows feed pass 2, so a bad one
      propagates; surfacing it lets the user fix the source.
    """
    warnings: list[QCWarning] = []
    for cue in output.cues:
        if not cue.body.strip():
            continue
        hedge = _HEDGE_RE.search(cue.plaintext)
        if hedge:
            warnings.append(
                QCWarning(
                    "gender",
                    f"unresolved pronoun hedge {hedge.group(0)!r}; commit to one",
                    cue.index,
                )
            )
    for row in bible.address:
        for field, who in (("other", row.get("to")), ("self", row.get("from"))):
            term = (row.get(field) or "").strip().lower()
            gender = _gender_of(bible, who)
            if not gender or (term not in _MALE_TERMS and term not in _FEMALE_TERMS):
                continue
            term_gender = "male" if term in _MALE_TERMS else "female"
            if term_gender != gender:
                warnings.append(
                    QCWarning(
                        "gender",
                        f"address {row.get('from')!r}->{row.get('to')!r}: "
                        f"{field}={row.get(field)!r} is {term_gender} but "
                        f"{who!r} is {gender}",
                    )
                )
    return warnings


def _gender_of(bible: Bible, name: str | None) -> str | None:
    """The recorded gender of the character named/aliased *name*, or ``None``."""
    if not name:
        return None
    for char in bible.characters:
        names = {char.get("name")} | set(char.get("aliases") or [])
        if name in names:
            gender = (char.get("gender") or "").strip().lower()
            if gender in ("male", "female"):
                return gender
    return None


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
