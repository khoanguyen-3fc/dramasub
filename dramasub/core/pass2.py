"""Pass 2 — chunk translation.

Translates the episode chunk by chunk. For each chunk the model is given a
sliding window (previous cues with their finalized translations, source-only
lookahead) and a bible excerpt filtered to just the characters, address pairs,
and glossary terms relevant to the chunk — never the whole bible.

Every chunk's result is validated: the returned index set must match exactly,
no value may be empty, inline ``{...}`` tags must be preserved, and no cue may
contain a script the target language doesn't use (guards against Chinese
idioms leaking from Chinese-trained models). On failure the chunk is retried
up to twice with an error hint; after that, individually valid cues are
salvaged and only the offending cues are left untranslated and reported — the
run never crashes mid-episode. Output is written with the mandatory
timing/count self-check and then QC'd.
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
from dramasub.core.lang import language_name
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


def _rewrap_doc(doc: SubtitleDoc, max_line_chars: int) -> None:
    """Re-wrap every cue's current body straight from the doc, right before a
    write. The last line of defense against a shipped 3+-line cue: idempotent,
    and independent of how the value reached the doc (main pass, cross-attempt
    salvage, rescue, or QC repair), so no path can leak a raw model line break
    into the output. Cues too long to fit two lines are left as-is (never
    truncated); those are a translation-length issue, not a wrapping one.
    """
    doc.apply(
        {c.index: subtitle.rewrap(c.body, max_line_chars)
         for c in doc.cues if c.body.strip()}
    )


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
    narrators = set((episode_context or {}).get("narrators", []))
    all_names = bible.character_names()
    honorific_note = _honorific_note(project)
    narration_note = _narration_note(narrators)
    temperature = project.temperature("pass2")
    max_line_chars = project.max_line_chars
    guide = project.load_guide()
    # Base dictionary underneath the show bible: the bible wins per source term.
    terms = _merge_terms(project.load_dictionary(), bible.glossary)

    translations: dict[int, str] = {}
    failures: list[ChunkFailure] = []

    for chunk in chunks:
        mapping = _translate_chunk(
            chunk,
            project=project,
            bible=bible,
            terms=terms,
            guide=guide,
            llm=llm,
            translations=translations,
            episode_cast=episode_cast,
            all_names=all_names,
            honorific_note=honorific_note,
            narration_note=narration_note,
            series_context=series_context,
            prev_summary=prev_summary,
            temperature=temperature,
            max_line_chars=max_line_chars,
        )
        chunk_ok, chunk_failed, reason = mapping
        translations.update(chunk_ok)
        if chunk_failed:
            failures.append(ChunkFailure(chunk.number, chunk_failed, reason))
            logger.error(
                "chunk %d: cues %s left untranslated (%s)", chunk.number, chunk_failed, reason
            )
        logger.info(
            "chunk %d/%d done (%d translated so far)",
            chunk.number,
            len(chunks),
            len(translations),
        )

    # Defensive final sweep: rewrap is idempotent, so normalizing once more at
    # apply time guarantees no code path can ship model line breaks, whatever
    # produced the value.
    translations = {
        i: subtitle.rewrap(body, max_line_chars) for i, body in translations.items()
    }
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
        _rewrap_doc(doc, max_line_chars)
        subtitle.save_verified(doc, out_path, source_ref)
        result.output_path = out_path
        result.qc_warnings = qc.run_qc(
            bible,
            source_ref,
            doc,
            max_line_chars=max_line_chars,
            target_language=project.target_language,
        )
        # QC-driven repair: re-request cues whose bible-glossary rendering is
        # missing (also recovers cues the model truncated mid-sentence, since
        # the dropped term forces a full regeneration). Accepted only when the
        # required rendering actually appears; otherwise the original stands.
        repaired = _repair_glossary_misses(
            project=project,
            bible=bible,
            doc=doc,
            source_ref=source_ref,
            qc_warnings=result.qc_warnings,
            llm=llm,
            temperature=temperature,
            max_line_chars=max_line_chars,
        )
        if repaired:
            translations.update(repaired)
            doc.apply(repaired)
        # Tighten pass: rewrap can only re-break, so a cue that is genuinely too
        # long for two lines (>~2x max_line_chars, or an unbreakable segment) is
        # re-requested more concisely. Reads the doc AFTER repair.
        tightened = _tighten_overlong(
            project=project,
            bible=bible,
            doc=doc,
            source_ref=source_ref,
            llm=llm,
            temperature=temperature,
            retry_temps=project.retry_temperatures,
            max_line_chars=max_line_chars,
        )
        if tightened:
            translations.update(tightened)
            doc.apply(tightened)
        if repaired or tightened:
            _rewrap_doc(doc, max_line_chars)
            subtitle.save_verified(doc, out_path, source_ref)
            result.qc_warnings = qc.run_qc(
                bible,
                source_ref,
                doc,
                max_line_chars=max_line_chars,
                target_language=project.target_language,
            )
            logger.info(
                "post-pass regen: %d repaired, %d tightened; %d warning(s) remain",
                len(repaired), len(tightened), len(result.qc_warnings),
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
    terms: list[dict[str, Any]],
    guide: str,
    llm: LLMClient,
    translations: dict[int, str],
    episode_cast: set[str],
    all_names: set[str],
    honorific_note: str,
    narration_note: str,
    series_context: str,
    prev_summary: str,
    temperature: float,
    max_line_chars: int,
) -> tuple[dict[int, str], list[int], str]:
    """Translate one chunk with per-cue validation and retries.

    Returns ``(translations, failed_indices, reason)``. After the retries are
    exhausted, individually valid cues from the best attempt are salvaged and
    only the offending cues are failed — one stubborn line no longer costs the
    whole chunk.
    """
    source_map = chunk.source_map()
    window_texts = [c.plaintext for c in (chunk.cues + chunk.prev + chunk.lookahead)]
    mentioned = {name for name in all_names if any(name in t for t in window_texts)}
    chunk_names = episode_cast | mentioned

    present_chars = bible.characters_for(chunk_names)
    canonical = {c.get("name") for c in present_chars if c.get("name")} | (
        episode_cast & all_names
    )
    address_rows = bible.address_rows_for(canonical)
    glossary_rows = _relevant_terms(terms, source_map.values())
    name_rows = bible.name_renderings(chunk_names)

    base_values = {
        "source_language": language_name(project.source_language),
        "target_language": language_name(project.target_language),
        "honorific_policy_note": honorific_note,
        "narration_note": narration_note,
        "style_note": project.style_guidance(),
        "guide": guide or "(none)",
        "series_context": series_context or "(none)",
        "prev_summary": prev_summary or "(start of episode)",
        "characters": _format_characters(present_chars),
        "names": _format_names(name_rows),
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
    # Valid cues are accumulated ACROSS attempts (first valid rendering wins —
    # earlier attempts run cooler and read better). Hot retries can fix the
    # flagged cue while newly leaking on another ("whack-a-mole"); the union
    # of attempts is often complete even when no single attempt is.
    salvaged: dict[int, str] = {}
    last_error: dict[int, str] = {}
    retry_temps = project.retry_temperatures
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = prompts.render("pass2", retry_hint=hint, **base_values)
        # Retries explore: at low temperature the model tends to regenerate the
        # same wrong tokens (e.g. a stubborn foreign idiom) despite the hint,
        # so retries run at the configured retry_temperatures (first attempt
        # uses the pass-2 temperature; an empty list keeps retries cool).
        if attempt == 1 or not retry_temps:
            attempt_temperature = temperature
        else:
            attempt_temperature = retry_temps[min(attempt - 2, len(retry_temps) - 1)]
        try:
            raw = llm.generate_json(prompt, temperature=attempt_temperature)
        except LLMError as exc:
            hint = _retry_hint(f"your previous reply could not be parsed as JSON ({exc})")
            logger.warning("chunk %d attempt %d: %s", chunk.number, attempt, exc)
            continue

        mapping, errors = _validate_chunk(raw, source_map, project.target_language)
        if None not in errors:  # structurally valid: bank this attempt's cues
            for index, body in mapping.items():
                salvaged.setdefault(index, body)
            for index, problem in errors.items():
                last_error[index] = problem
        if errors:
            logger.warning(
                "chunk %d attempt %d invalid: %s",
                chunk.number, attempt, _describe(errors),
            )
        missing = [i for i in source_map if i not in salvaged]
        if not missing:
            if attempt > 1 or errors:
                logger.info(
                    "chunk %d: all cues covered after %d attempt(s) via "
                    "cross-attempt salvage", chunk.number, attempt,
                )
            break  # every cue covered by some attempt — done, even mid-retry
        hint = _retry_hint(_describe(errors) if errors else "reply unusable")

    failed = sorted(i for i in source_map if i not in salvaged)
    if failed:
        # Rescue pass: re-request ONLY the cues that never validated, with the
        # chunk's salvaged translations as context. The smaller framing often
        # escapes a leak the full-chunk regeneration kept re-triggering, and
        # costs a few cues instead of a whole chunk.
        _rescue_failed(
            chunk=chunk,
            base_values=base_values,
            source_map=source_map,
            salvaged=salvaged,
            last_error=last_error,
            translations=translations,
            project=project,
            llm=llm,
            temperature=temperature,
            retry_temps=retry_temps,
        )
        failed = sorted(i for i in source_map if i not in salvaged)

    reason = "; ".join(
        dict.fromkeys(last_error.get(i, "chunk validation failed") for i in failed)
    )
    # Wrapping is decided by the target text's width, not inherited from the
    # source: drop the model's line breaks and re-wrap each cue deterministically
    # to <=2 balanced lines. (Korean is denser and differently spaced, so models
    # tend to fragment the Vietnamese to mirror the source — this prevents that.)
    wrapped = {i: subtitle.rewrap(body, max_line_chars) for i, body in salvaged.items()}
    return wrapped, failed, reason


RESCUE_ATTEMPTS = 2
TIGHTEN_ATTEMPTS = 3  # over-long cues are few; a couple of hotter retries help


def _repair_glossary_misses(
    *,
    project: Any,
    bible: Bible,
    doc: SubtitleDoc,
    source_ref: SubtitleDoc,
    qc_warnings: list[QCWarning],
    llm: LLMClient,
    temperature: float,
    max_line_chars: int,
) -> dict[int, str]:
    """Re-request cues QC flagged for missing glossary renderings.

    One bounded attempt per group of nearby cues, with the required
    ``term -> rendering`` pairs given as an explicit correction. A repair is
    accepted only if it validates AND actually contains every required
    rendering; otherwise the original translation stands. Returns the
    accepted (already wrapped) replacements.
    """
    flagged = sorted(
        {w.index for w in qc_warnings if w.kind == "glossary" and w.index is not None}
    )
    if not flagged:
        return {}
    entries = [
        (e["source"], e["target"])
        for e in bible.glossary
        if e.get("source") and e.get("target")
    ]
    required: dict[int, list[tuple[str, str]]] = {}
    for i in flagged:
        src_text = source_ref.cue(i).plaintext
        out_text = doc.cue(i).plaintext.lower()
        need = [(t, r) for t, r in entries if t in src_text and r.lower() not in out_text]
        if need:
            required[i] = need
    if not required:
        return {}
    logger.info(
        "QC repair: re-requesting %d cue(s) with missing glossary renderings",
        len(required),
    )
    accepted: dict[int, str] = {}
    guide = project.load_guide()
    for group in _group_indices(sorted(required), gap=2):
        group_map = {i: source_ref.cue(i).body for i in group}
        gloss_rows = [
            {"source": t, "target": r, "note": "required"}
            for i in group
            for t, r in required[i]
        ]
        prev_ids = [j for j in range(group[0] - 4, group[0]) if j >= 0]
        look_ids = [
            j for j in range(group[-1] + 1, group[-1] + 3) if j < len(source_ref.cues)
        ]
        values = {
            "source_language": language_name(project.source_language),
            "target_language": language_name(project.target_language),
            "honorific_policy_note": _honorific_note(project),
            "narration_note": "(none)",
            "style_note": project.style_guidance(),
            "guide": guide or "(none)",
            "series_context": "(none)",
            "prev_summary": "(mid-episode repair)",
            "characters": "(none)",
            "names": "(none)",
            "address": "(none)",
            "glossary": _format_glossary(gloss_rows),
            "prev_context": _format_prev(
                [source_ref.cue(j) for j in prev_ids],
                {j: doc.cue(j).body for j in prev_ids},
            ),
            "lookahead": _format_lookahead([source_ref.cue(j) for j in look_ids]),
            "source_json": json.dumps(
                {str(i): t for i, t in group_map.items()}, ensure_ascii=False, indent=2
            ),
            "max_line_chars": max_line_chars,
        }
        hint = _retry_hint(
            "these lines previously omitted required glossary renderings: "
            + "; ".join(
                f"index {i} must contain {r!r} for {t!r}"
                for i in group
                for t, r in required[i]
            )
        )
        prompt = prompts.render("pass2", retry_hint=hint, **values)
        try:
            raw = llm.generate_json(prompt, temperature=temperature)
        except LLMError as exc:
            logger.warning("QC repair request failed: %s", exc)
            continue
        mapping, _errors = _validate_chunk(raw, group_map, project.target_language)
        for i, value in mapping.items():
            if all(r.lower() in value.lower() for _, r in required[i]):
                accepted[i] = subtitle.rewrap(value, max_line_chars)
            else:
                logger.info(
                    "QC repair: cue %d still missing its rendering; keeping original", i
                )
    return accepted


def _tighten_overlong(
    *,
    project: Any,
    bible: Bible,
    doc: SubtitleDoc,
    source_ref: SubtitleDoc,
    llm: LLMClient,
    temperature: float,
    retry_temps: list[float],
    max_line_chars: int,
) -> dict[int, str]:
    """Re-request a shorter rendering for cues that cannot fit two lines.

    ``rewrap`` only re-breaks text; when a translation is simply too long — over
    the two-line ``max_line_chars`` budget, or with an unbreakable segment that
    forces a line past the limit — no wrapping can satisfy the rule, so the model
    is asked to say it more concisely. Accepted only if the reply validates AND
    now fits; otherwise the original stands (never truncated). Returns the
    accepted, already-wrapped replacements.
    """
    def fits(body: str) -> bool:
        lines = subtitle.rendered_lines(subtitle.rewrap(body, max_line_chars))
        return len(lines) <= 2 and all(
            subtitle.display_len(ln) <= max_line_chars for ln in lines
        )

    overlong = [c.index for c in doc.cues if c.body.strip() and not fits(c.body)]
    if not overlong:
        return {}
    logger.info(
        "tighten: %d cue(s) over the two-line budget: %s", len(overlong), overlong
    )

    entries = [
        (e["source"], e["target"])
        for e in bible.glossary
        if e.get("source") and e.get("target")
    ]
    guide = project.load_guide()
    budget = max_line_chars * 2
    accepted: dict[int, str] = {}
    for i in overlong:
        src_body = source_ref.cue(i).body
        src_text = source_ref.cue(i).plaintext
        gloss_rows = [{"source": t, "target": r} for t, r in entries if t in src_text]
        prev_ids = [j for j in range(i - 4, i) if j >= 0]
        look_ids = [j for j in range(i + 1, i + 3) if j < len(source_ref.cues)]
        values = {
            "source_language": language_name(project.source_language),
            "target_language": language_name(project.target_language),
            "honorific_policy_note": _honorific_note(project),
            "narration_note": "(none)",
            "style_note": project.style_guidance(),
            "guide": guide or "(none)",
            "series_context": "(none)",
            "prev_summary": "(mid-episode tighten)",
            "characters": "(none)",
            "names": "(none)",
            "address": "(none)",
            "glossary": _format_glossary(gloss_rows),
            "prev_context": _format_prev(
                [source_ref.cue(j) for j in prev_ids],
                {j: doc.cue(j).body for j in prev_ids},
            ),
            "lookahead": _format_lookahead([source_ref.cue(j) for j in look_ids]),
            "source_json": json.dumps({str(i): src_body}, ensure_ascii=False, indent=2),
            "max_line_chars": max_line_chars,
        }
        current = doc.cue(i).plaintext.replace("\n", " ")
        for attempt in range(1, TIGHTEN_ATTEMPTS + 1):
            temp = (temperature if attempt == 1 or not retry_temps
                    else retry_temps[min(attempt - 2, len(retry_temps) - 1)])
            harder = (" The previous rewrite was still too long — cut harder."
                      if attempt > 1 else "")
            hint = _retry_hint(
                f"index {i} is too long to fit two subtitle lines of {max_line_chars} "
                f"characters (about {budget} total). Your previous translation was: "
                f"{current!r}. Rewrite it MORE CONCISELY with the same meaning and "
                f"register — cut filler, not content — and keep any {{...}} tags.{harder}"
            )
            prompt = prompts.render("pass2", retry_hint=hint, **values)
            try:
                raw = llm.generate_json(prompt, temperature=temp)
            except LLMError as exc:
                logger.warning("tighten request failed for cue %d: %s", i, exc)
                continue
            mapping, _errors = _validate_chunk(raw, {i: src_body}, project.target_language)
            value = mapping.get(i)
            if value is None:
                continue
            if fits(value):
                accepted[i] = subtitle.rewrap(value, max_line_chars)
                break
            current = " ".join(subtitle.rendered_lines(value))  # feed it back, try again
        else:
            logger.info(
                "tighten: cue %d still over-long after %d attempt(s); keeping original",
                i, TIGHTEN_ATTEMPTS,
            )
    return accepted


def _group_indices(indices: list[int], gap: int) -> list[list[int]]:
    groups = [[indices[0]]]
    for index in indices[1:]:
        if index - groups[-1][-1] <= gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _rescue_failed(
    *,
    chunk: Chunk,
    base_values: dict[str, Any],
    source_map: dict[int, str],
    salvaged: dict[int, str],
    last_error: dict[int, str],
    translations: dict[int, str],
    project: Any,
    llm: LLMClient,
    temperature: float,
    retry_temps: list[float],
) -> None:
    """Re-request the never-validated cues, banking recoveries in place.

    First re-requests all failed cues together (already-salvaged cues shown as
    translated context so recoveries stay consistent with their neighbors),
    then falls back to translating any still-missing cue on its own — a
    single-index request cannot drop its own index — so no line is left
    untranslated.
    """
    rescue_map = {i: source_map[i] for i in source_map if i not in salvaged}
    logger.info(
        "chunk %d: rescue pass for cue(s) %s", chunk.number, sorted(rescue_map)
    )
    values = dict(base_values)
    values["source_json"] = json.dumps(
        {str(i): t for i, t in rescue_map.items()}, ensure_ascii=False, indent=2
    )
    context_cues = chunk.prev + [c for c in chunk.cues if c.index in salvaged]
    values["prev_context"] = _format_prev(context_cues, {**translations, **salvaged})
    hint = _retry_hint(
        _describe({i: last_error[i] for i in rescue_map if i in last_error})
        or "previous attempts failed for these lines"
    )
    for attempt in range(1, RESCUE_ATTEMPTS + 1):
        if attempt == 1 or not retry_temps:
            attempt_temperature = temperature
        else:
            attempt_temperature = retry_temps[min(attempt - 2, len(retry_temps) - 1)]
        prompt = prompts.render("pass2", retry_hint=hint, **values)
        try:
            raw = llm.generate_json(prompt, temperature=attempt_temperature)
        except LLMError as exc:
            logger.warning("chunk %d rescue attempt %d: %s", chunk.number, attempt, exc)
            continue
        mapping, errors = _validate_chunk(raw, rescue_map, project.target_language)
        if None not in errors:
            for index, body in mapping.items():
                salvaged.setdefault(index, body)
            for index, problem in errors.items():
                last_error[index] = problem
        missing = [i for i in rescue_map if i not in salvaged]
        if not missing:
            logger.info("chunk %d: rescue recovered all cue(s)", chunk.number)
            return
        if errors:
            logger.warning(
                "chunk %d rescue attempt %d invalid: %s",
                chunk.number, attempt, _describe(errors),
            )
        hint = _retry_hint(_describe(errors) if errors else "reply unusable")

    # Last-resort single-cue pass: a one-index request cannot drop its own
    # index, so each cue still missing is translated on its own. Guarantees no
    # line is left untranslated, at the cost of one call per stubborn cue.
    for i in [j for j in rescue_map if j not in salvaged]:
        vals = dict(base_values)
        vals["source_json"] = json.dumps(
            {str(i): source_map[i]}, ensure_ascii=False, indent=2
        )
        vals["prev_context"] = _format_prev(context_cues, {**translations, **salvaged})
        prompt = prompts.render(
            "pass2",
            retry_hint=_retry_hint("translate ONLY this one line; keep its index"),
            **vals,
        )
        try:
            raw = llm.generate_json(prompt, temperature=temperature)
        except LLMError as exc:
            logger.warning("chunk %d single-cue rescue of %d failed: %s", chunk.number, i, exc)
            continue
        mapping, _errors = _validate_chunk(raw, {i: source_map[i]}, project.target_language)
        if i in mapping:
            salvaged.setdefault(i, mapping[i])
    still = [i for i in rescue_map if i not in salvaged]
    if not still:
        logger.info("chunk %d: all cue(s) recovered", chunk.number)
    else:
        logger.warning(
            "chunk %d: %d cue(s) unrecovered after single-cue pass: %s",
            chunk.number, len(still), still,
        )


def _validate_chunk(
    raw: Any, source_map: dict[int, str], target_language: str
) -> tuple[dict[int, str], dict[int | None, str]]:
    """Validate a chunk response per cue.

    Returns ``(mapping, errors)``: *mapping* holds the individually valid
    cues; *errors* maps a cue index (or ``None`` for chunk-level problems,
    which make the whole reply unusable) to a retry hint. An empty *errors*
    means the chunk validated completely.
    """
    raw = _unwrap(raw, source_map)
    if not isinstance(raw, dict):
        return {}, {None: "reply must be a JSON object mapping each index to a translation"}

    got = {str(k).strip(): v for k, v in raw.items()}
    # A missing or extra index is NOT fatal: bank every expected cue that is
    # present and valid, flag each missing one per-index (so retries and the
    # rescue pass target only it), and ignore any unexpected extras. A single
    # dropped index must never discard the whole reply's good translations.
    mapping: dict[int, str] = {}
    errors: dict[int | None, str] = {}
    for index, body in source_map.items():
        if str(index) not in got:
            errors[index] = f"index {index} is missing from the reply; include it"
            continue
        value = got[str(index)]
        if not isinstance(value, str) or not value.strip():
            errors[index] = f"index {index} has an empty or non-string value"
            continue
        value = subtitle.normalize_linebreaks(value)
        preserve = preservation_error(body, value)
        if preserve is not None:
            errors[index] = f"index {index}: {preserve}"
            continue
        leak = qc.foreign_script(value, target_language)
        if leak:
            name = language_name(target_language)
            errors[index] = (
                f"index {index} contains {leak!r}, which is not {name}; "
                f"rewrite that line entirely in {name}, expressing any "
                "idiom with a native equivalent"
            )
            continue
        mapping[index] = value
    return mapping, errors


def _describe(errors: dict[int | None, str]) -> str:
    seen: list[str] = []
    for key in sorted(errors, key=lambda x: (x is None, x if x is not None else -1)):
        if errors[key] not in seen:
            seen.append(errors[key])
    return "; ".join(seen)


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
        target_language=language_name(project.target_language),
        prev_summary=prev_summary or "(this is the first episode)",
        dialogue="\n".join(lines),
    )
    try:
        return llm.generate(prompt, temperature=project.temperature("summary")).strip()
    except LLMError as exc:
        logger.warning("could not generate episode summary: %s", exc)
        return ""


def _narration_note(narrators: set[str]) -> str:
    note = (
        "Some lines may be narration, voiceover, or commentary spoken to the "
        "audience rather than to another character. Render those in the third "
        "person and do not invent a second-person pronoun for the speaker."
    )
    if narrators:
        note += " Known narrators/commentators this episode: " + ", ".join(sorted(narrators)) + "."
    return note


def _honorific_note(project: Any) -> str:
    template = _HONORIFIC_NOTES.get(project.honorific_policy, _HONORIFIC_NOTES["translate"])
    return template.format(target=language_name(project.target_language))


def _merge_terms(
    dictionary: list[dict[str, Any]], glossary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Base dictionary overlaid by the show glossary (glossary wins per term)."""
    merged: dict[str, dict[str, Any]] = {}
    for entry in list(dictionary) + list(glossary):
        if entry.get("source") and entry.get("target"):
            merged[entry["source"]] = entry
    return list(merged.values())


def _relevant_terms(
    terms: list[dict[str, Any]], source_bodies: Any
) -> list[dict[str, Any]]:
    joined = "\n".join(source_bodies)
    return [e for e in terms if e["source"] in joined]


def _format_characters(chars: list[dict[str, Any]]) -> str:
    if not chars:
        return "(no specific characters identified for this chunk)"
    out = []
    for char in chars:
        line = f"- {char.get('name', '?')}"
        gender = (char.get("gender") or "").lower()
        if gender in ("male", "female"):
            line += f" [{'nam' if gender == 'male' else 'nữ'}]"
        if char.get("role"):
            line += f" — {char['role']}"
        if char.get("aliases"):
            line += f" (aka {', '.join(char['aliases'])})"
        out.append(line)
    return "\n".join(out)


def _format_names(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "(no frozen name renderings yet; keep any names you use consistent)"
    return "\n".join(f"- {src} = {tgt}" for src, tgt in rows)


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
