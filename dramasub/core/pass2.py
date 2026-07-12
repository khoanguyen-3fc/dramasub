"""Pass 2 — chunk translation.

Translates the episode chunk by chunk. For each chunk the model is given a
sliding window (previous cues with their finalized translations, source-only
lookahead) and a bible excerpt filtered to just the characters, address pairs,
and glossary terms relevant to the chunk — never the whole bible.

Every chunk's result is validated: the returned index set must match exactly,
no value may be empty, and inline ``{...}`` tags / ``\\N`` line breaks must be
preserved. On failure the chunk is retried up to twice with an error hint; if
it still fails those cues are left untranslated and reported — the run never
crashes mid-episode. Output is written with the mandatory timing/count
self-check and then QC'd.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dramasub.core import chunker, prompts, qc, subtitle
from dramasub.core.bible import Bible
from dramasub.core.chunker import Chunk
from dramasub.core.errors import LLMError
from dramasub.core.llm import LLMClient
from dramasub.core.qc import QCWarning
from dramasub.core.subtitle import SubtitleDoc, preservation_error

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3  # 1 initial + 2 retries (AGENTS.md: retry a chunk up to 2 times)
SUMMARY_CHAR_BUDGET = 12000

_HONORIFIC_NOTES = {
    "translate": (
        "Render honorifics and address terms as natural {target} equivalents; "
        "do not leave source-language honorifics romanized unless the glossary "
        "says otherwise."
    ),
    "keep_romanized": (
        "Keep source-language honorifics/address terms romanized (e.g. oppa, "
        "sunbae). Still translate the surrounding dialogue into {target}."
    ),
}


@dataclass
class ChunkFailure:
    chunk_number: int
    indices: list[int]
    reason: str


@dataclass
class TranslateResult:
    episode: int
    output_path: Path | None
    translated_count: int
    failed_indices: list[int]
    failures: list[ChunkFailure] = field(default_factory=list)
    qc_warnings: list[QCWarning] = field(default_factory=list)
    summary: str = ""

    @property
    def total_cues(self) -> int:
        return self.translated_count + len(self.failed_indices)


def translate_episode(
    project: Any,
    bible: Bible,
    doc: SubtitleDoc,
    *,
    episode: int,
    llm: LLMClient,
    episode_context: dict[str, Any] | None = None,
    series_context: str = "",
    prev_summary: str = "",
    write: bool = True,
    generate_summary: bool = True,
) -> TranslateResult:
    """Translate *doc* and (optionally) write ``output.srt`` and ``summary.txt``.

    *doc* is mutated in place with the translations. The caller supplies the
    same loaded source doc it parsed; timing is re-verified against a pristine
    reload of the source file on write.
    """
    indices = doc.translatable_indices()
    if not indices:
        logger.warning("episode %d has no translatable cues; nothing to do", episode)
        return TranslateResult(episode, None, 0, [])

    cues_seq = [doc.cue(i) for i in indices]
    chunks = chunker.plan(
        cues_seq,
        size=project.chunk_setting("size"),
        prev_window=project.chunk_setting("prev_window"),
        lookahead=project.chunk_setting("lookahead"),
    )

    episode_cast = set((episode_context or {}).get("characters_present", []))
    all_names = bible.character_names()
    honorific_note = _honorific_note(project)
    temperature = project.temperature("pass2")
    max_line_chars = project.max_line_chars

    translations: dict[int, str] = {}
    failures: list[ChunkFailure] = []

    for chunk in chunks:
        mapping = _translate_chunk(
            chunk,
            project=project,
            bible=bible,
            llm=llm,
            translations=translations,
            episode_cast=episode_cast,
            all_names=all_names,
            honorific_note=honorific_note,
            series_context=series_context,
            prev_summary=prev_summary,
            temperature=temperature,
            max_line_chars=max_line_chars,
        )
        if mapping is None:
            failures.append(
                ChunkFailure(chunk.number, chunk.indices, "validation failed after retries")
            )
            logger.error("chunk %d failed; cues %s left untranslated", chunk.number, chunk.indices)
        else:
            translations.update(mapping)
        logger.info(
            "chunk %d/%d done (%d translated so far)",
            chunk.number,
            len(chunks),
            len(translations),
        )

    doc.apply(translations)

    failed_indices = sorted(i for f in failures for i in f.indices)
    result = TranslateResult(
        episode=episode,
        output_path=None,
        translated_count=len(translations),
        failed_indices=failed_indices,
        failures=failures,
    )

    if write:
        source_ref = subtitle.load(doc.path)
        out_path = project.episode_dir(episode) / f"output{doc.path.suffix.lower()}"
        subtitle.save_verified(doc, out_path, source_ref)
        result.output_path = out_path
        result.qc_warnings = qc.run_qc(
            bible, source_ref, doc, max_line_chars=max_line_chars
        )

    if generate_summary:
        result.summary = _generate_summary(project, doc, llm, prev_summary)
        if write and result.summary:
            project.episode_summary(episode).write_text(result.summary + "\n", encoding="utf-8")

    logger.info(
        "episode %d: %d/%d cues translated, %d failed, %d QC warnings",
        episode,
        result.translated_count,
        result.total_cues,
        len(failed_indices),
        len(result.qc_warnings),
    )
    return result


def _translate_chunk(
    chunk: Chunk,
    *,
    project: Any,
    bible: Bible,
    llm: LLMClient,
    translations: dict[int, str],
    episode_cast: set[str],
    all_names: set[str],
    honorific_note: str,
    series_context: str,
    prev_summary: str,
    temperature: float,
    max_line_chars: int,
) -> dict[int, str] | None:
    """Translate one chunk with validation + retries. Returns the mapping or None."""
    source_map = chunk.source_map()
    window_texts = [c.plaintext for c in (chunk.cues + chunk.prev + chunk.lookahead)]
    mentioned = {name for name in all_names if any(name in t for t in window_texts)}
    chunk_names = episode_cast | mentioned

    present_chars = bible.characters_for(chunk_names)
    canonical = {c.get("name") for c in present_chars if c.get("name")} | (
        episode_cast & all_names
    )
    address_rows = bible.address_rows_for(canonical)
    glossary_rows = _relevant_glossary(bible, source_map.values())

    base_values = {
        "source_language": project.source_language,
        "target_language": project.target_language,
        "honorific_policy_note": honorific_note,
        "series_context": series_context or "(none)",
        "prev_summary": prev_summary or "(start of episode)",
        "characters": _format_characters(present_chars),
        "address": _format_address(address_rows),
        "glossary": _format_glossary(glossary_rows),
        "prev_context": _format_prev(chunk.prev, translations),
        "lookahead": _format_lookahead(chunk.lookahead),
        "source_json": json.dumps(
            {str(i): t for i, t in source_map.items()}, ensure_ascii=False, indent=2
        ),
        "max_line_chars": max_line_chars,
    }

    hint = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = prompts.render("pass2", retry_hint=hint, **base_values)
        try:
            raw = llm.generate_json(prompt, temperature=temperature)
        except LLMError as exc:
            hint = _retry_hint(f"your previous reply could not be parsed as JSON ({exc})")
            logger.warning("chunk %d attempt %d: %s", chunk.number, attempt, exc)
            continue

        mapping, error = _validate_chunk(raw, source_map)
        if error is None:
            return mapping
        hint = _retry_hint(error)
        logger.warning("chunk %d attempt %d invalid: %s", chunk.number, attempt, error)

    return None


def _validate_chunk(
    raw: Any, source_map: dict[int, str]
) -> tuple[dict[int, str] | None, str | None]:
    """Validate a chunk response. Returns ``(mapping, None)`` or ``(None, hint)``."""
    raw = _unwrap(raw, source_map)
    if not isinstance(raw, dict):
        return None, "reply must be a JSON object mapping each index to a translation"

    got = {str(k).strip(): v for k, v in raw.items()}
    expected = {str(i) for i in source_map}
    missing = expected - got.keys()
    extra = got.keys() - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing indices {sorted(missing)}")
        if extra:
            parts.append(f"unexpected indices {sorted(extra)}")
        return None, (
            "the reply's indices must match the input exactly: "
            + "; ".join(parts)
        )

    mapping: dict[int, str] = {}
    for index, body in source_map.items():
        value = got[str(index)]
        if not isinstance(value, str) or not value.strip():
            return None, f"index {index} has an empty or non-string value"
        value = subtitle.normalize_linebreaks(value)
        preserve = preservation_error(body, value)
        if preserve is not None:
            return None, f"index {index}: {preserve}"
        mapping[index] = value
    return mapping, None


def _unwrap(raw: Any, source_map: dict[int, str]) -> Any:
    """Unwrap a single-key envelope like ``{"translations": {...}}``."""
    if isinstance(raw, dict) and len(raw) == 1:
        (only_value,) = raw.values()
        if isinstance(only_value, dict):
            expected = {str(i) for i in source_map}
            if expected & {str(k) for k in only_value}:
                return only_value
    return raw


def _generate_summary(
    project: Any, doc: SubtitleDoc, llm: LLMClient, prev_summary: str
) -> str:
    lines: list[str] = []
    total = 0
    for cue in doc.cues:
        if not cue.is_translatable:
            continue
        text = cue.plaintext.replace("\n", " ").strip()
        if not text:
            continue
        total += len(text) + 1
        if total > SUMMARY_CHAR_BUDGET:
            lines.append("[... episode continues ...]")
            break
        lines.append(text)
    if not lines:
        return ""
    prompt = prompts.render(
        "summary",
        target_language=project.target_language,
        prev_summary=prev_summary or "(this is the first episode)",
        dialogue="\n".join(lines),
    )
    try:
        return llm.generate(prompt, temperature=project.temperature("summary")).strip()
    except LLMError as exc:
        logger.warning("could not generate episode summary: %s", exc)
        return ""


def _honorific_note(project: Any) -> str:
    template = _HONORIFIC_NOTES.get(project.honorific_policy, _HONORIFIC_NOTES["translate"])
    return template.format(target=project.target_language)


def _relevant_glossary(bible: Bible, source_bodies: Any) -> list[dict[str, Any]]:
    joined = "\n".join(source_bodies)
    return [
        e
        for e in bible.glossary
        if e.get("source") and e.get("target") and e["source"] in joined
    ]


def _format_characters(chars: list[dict[str, Any]]) -> str:
    if not chars:
        return "(no specific characters identified for this chunk)"
    out = []
    for char in chars:
        line = f"- {char.get('name', '?')}"
        if char.get("role"):
            line += f" — {char['role']}"
        if char.get("aliases"):
            line += f" (aka {', '.join(char['aliases'])})"
        out.append(line)
    return "\n".join(out)


def _format_address(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no address pairs recorded for this scene)"
    out = []
    for row in rows:
        line = (
            f"- when {row.get('from')} speaks to {row.get('to')}: "
            f"self = {row.get('self')}, addresses them as {row.get('other')}"
        )
        if row.get("note"):
            line += f" ({row['note']})"
        out.append(line)
    return "\n".join(out)


def _format_glossary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no glossary terms apply to this chunk)"
    return "\n".join(
        f"- {e['source']} -> {e['target']}"
        + (f" ({e['note']})" if e.get("note") else "")
        for e in rows
    )


def _format_prev(prev: list[Any], translations: dict[int, str]) -> str:
    if not prev:
        return "(this is the start of the episode)"
    out = []
    for cue in prev:
        translated = translations.get(cue.index)
        src = cue.plaintext.replace("\n", " ")
        if translated:
            out.append(f"{cue.index}: {src}  =>  {translated}")
        else:
            out.append(f"{cue.index}: {src}  (not translated)")
    return "\n".join(out)


def _format_lookahead(lookahead: list[Any]) -> str:
    if not lookahead:
        return "(none)"
    return "\n".join(
        f"{cue.index}: {cue.plaintext.replace(chr(10), ' ')}" for cue in lookahead
    )


def _retry_hint(message: str) -> str:
    return (
        "\n## Correction needed\n"
        f"Your previous attempt was rejected: {message}. "
        "Return corrected JSON with exactly the required indices."
    )
