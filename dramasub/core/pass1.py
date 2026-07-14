"""Pass 1 — episode context extraction.

Reads the episode's dialogue and extracts structured context: who appears, who
speaks to whom (and at what register), relationship/speech-level changes, and
new terms. The result is written to the episode's ``context.yaml`` and its
``proposed_updates`` are auto-applied to the bible (append/update only, logged
to ``change_log`` so the user can review and revert).

Long episodes are analyzed in segments so we never blow past ``num_ctx``; the
per-segment results are merged into one episode context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from dramasub.core import prompts
from dramasub.core._yaml import write_yaml
from dramasub.core.bible import Bible
from dramasub.core.lang import language_name
from dramasub.core.llm import LLMClient, LLMError
from dramasub.core.subtitle import SubtitleDoc

logger = logging.getLogger(__name__)

# Cues per pass-1 LLM call. Small enough that the dialogue plus bible/series
# context stays well inside a 16k context window.
SEGMENT_CUES = 150
# A segment whose JSON answer overflows num_ctx comes back truncated (and thus
# unparseable). Retry it with a wider context window before giving up — a
# transient truncation must not abort the whole episode's pass 1.
PASS1_ATTEMPTS = 3

_HONORIFIC_NOTES = {
    "translate": (
        "Render honorifics and address terms as natural {target} equivalents; "
        "do not leave source-language honorifics romanized."
    ),
    "keep_romanized": (
        "Keep source-language honorifics/address terms romanized (e.g. oppa, "
        "sunbae) rather than translating them."
    ),
}


@dataclass
class Pass1Result:
    """Structured episode context plus the bible changes it triggered."""

    episode: int
    context: dict[str, Any]
    applied_updates: list[str] = field(default_factory=list)

    @property
    def characters_present(self) -> list[str]:
        return self.context.get("characters_present", [])


def extract_context(
    project: Any,
    bible: Bible,
    doc: SubtitleDoc,
    *,
    episode: int,
    llm: LLMClient,
    series_context: str = "",
    prev_summary: str = "",
    segment_cues: int = SEGMENT_CUES,
    write: bool = True,
) -> Pass1Result:
    """Run pass 1 over *doc*, update *bible*, and persist ``context.yaml``.

    The caller owns saving the bible; this function mutates it in memory and
    returns the change notes so the CLI can report and persist them.
    """
    indices = doc.translatable_indices()
    if not indices:
        logger.warning("episode %d has no translatable cues", episode)

    known_characters = _format_characters(bible)
    known_address = _format_address(bible)
    # Show the base dictionary alongside the show glossary so pass 1 doesn't
    # re-propose universal terms (bible wins per source term).
    dictionary = project.load_dictionary()
    dictionary_sources = {e["source"] for e in dictionary}
    known_terms: dict[str, dict[str, Any]] = {}
    for entry in dictionary + bible.glossary:
        if entry.get("source") and entry.get("target"):
            known_terms[entry["source"]] = entry
    known_glossary = _format_glossary(list(known_terms.values()))
    honorific_note = _honorific_note(project)
    temperature = project.temperature("pass1")

    merged = _empty_context(episode)
    segments = _segments(indices, segment_cues)
    for seg_no, seg in enumerate(segments, start=1):
        dialogue = _format_dialogue(doc, seg)
        label = f"{seg_no}/{len(segments)} (cues {seg[0]}-{seg[-1]})"
        prompt = prompts.render(
            "pass1",
            source_language=language_name(project.source_language),
            target_language=language_name(project.target_language),
            segment_label=label,
            series_context=series_context or "(none provided)",
            prev_summary=prev_summary or "(this is the first analyzed segment)",
            known_characters=known_characters,
            known_address=known_address,
            known_glossary=known_glossary,
            honorific_policy_note=honorific_note,
            style_note=project.style_guidance(),
            episode=episode,
            dialogue=dialogue,
        )
        logger.info("pass 1: analyzing segment %s", label)
        raw = None
        for attempt in range(1, PASS1_ATTEMPTS + 1):
            # A parse failure here means the JSON was truncated for lack of room,
            # so widen the context window on retry rather than just re-sampling.
            ctx = project.num_ctx if attempt == 1 else project.num_ctx * 2
            try:
                raw = llm.generate_json(prompt, temperature=temperature, num_ctx=ctx)
                break
            except LLMError as exc:
                logger.warning(
                    "pass 1: segment %s attempt %d/%d failed (%s); retrying with a "
                    "wider context window", label, attempt, PASS1_ATTEMPTS, exc,
                )
        if raw is None:
            raise LLMError(
                f"pass 1: segment {label} still unparseable after {PASS1_ATTEMPTS} "
                "attempts (model JSON kept truncating; raise num_ctx in project.yaml)"
            )
        normalized = _normalize_context(raw, episode)
        _merge_into(merged, normalized)

    # Never auto-apply a glossary proposal the base dictionary already covers:
    # the bible outranks the dictionary, so an auto-proposal would permanently
    # shadow the curated default. Overriding a default is a human decision,
    # made by adding the term to bible.yaml by hand.
    proposals = merged["proposed_updates"].get("glossary", [])
    kept = [
        t for t in proposals
        if not _covered_by(t.get("source") or "", dictionary_sources)
    ]
    if len(kept) != len(proposals):
        skipped = [t.get("source", "?") for t in proposals if t not in kept]
        logger.info(
            "pass 1: skipped %d glossary proposal(s) already covered by the "
            "base dictionary: %s", len(skipped), ", ".join(skipped),
        )
    merged["proposed_updates"]["glossary"] = kept

    applied = bible.apply_updates(episode, merged["proposed_updates"])
    merged["applied_updates"] = applied

    if write:
        write_yaml(project.episode_context(episode), merged)
        logger.info(
            "pass 1 complete for episode %d: %d characters, %d bible updates",
            episode,
            len(merged["characters_present"]),
            len(applied),
        )
    return Pass1Result(episode=episode, context=merged, applied_updates=applied)


def _covered_by(source: str, dictionary_sources: set[str]) -> bool:
    """True when a proposed term is already covered by the base dictionary.

    Covers exact matches and short suffixed variants (e.g. an honorific
    particle appended to a dictionary term), so "<term>님"-style proposals
    don't shadow the curated default either.
    """
    if not source:
        return False
    if source in dictionary_sources:
        return True
    return any(
        source.startswith(term) and len(source) - len(term) <= 2
        for term in dictionary_sources
    )


def _honorific_note(project: Any) -> str:
    template = _HONORIFIC_NOTES.get(project.honorific_policy, _HONORIFIC_NOTES["translate"])
    return template.format(target=language_name(project.target_language))


def _segments(indices: list[int], segment_cues: int) -> list[list[int]]:
    if not indices:
        return []
    return [indices[i:i + segment_cues] for i in range(0, len(indices), segment_cues)]


def _format_dialogue(doc: SubtitleDoc, indices: list[int]) -> str:
    lines = []
    for i in indices:
        text = doc.cue(i).plaintext.replace("\n", " ").strip()
        lines.append(f"{i}: {text}")
    return "\n".join(lines)


def _format_characters(bible: Bible) -> str:
    if not bible.characters:
        return "(none known yet)"
    out = []
    for char in bible.characters:
        aliases = ", ".join(char.get("aliases", []) or [])
        role = char.get("role", "")
        line = f"- {char.get('name', '?')}"
        if role:
            line += f" — {role}"
        if aliases:
            line += f" (aka {aliases})"
        out.append(line)
    return "\n".join(out)


def _format_address(bible: Bible) -> str:
    if not bible.address:
        return "(none known yet)"
    out = []
    for row in bible.address:
        note = row.get("note", "")
        line = (
            f"- {row.get('from')} -> {row.get('to')}: self={row.get('self')}, "
            f"other={row.get('other')}"
        )
        if row.get("since_episode") is not None:
            line += f" (since ep {row.get('since_episode')})"
        if note:
            line += f" — {note}"
        out.append(line)
    return "\n".join(out)


def _format_glossary(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "(none known yet)"
    out = []
    for term in entries:
        line = f"- {term.get('source')} = {term.get('target')}"
        if term.get("note"):
            line += f" ({term.get('note')})"
        out.append(line)
    return "\n".join(out)


def _empty_context(episode: int) -> dict[str, Any]:
    return {
        "episode": episode,
        "characters_present": [],
        "speaker_pairs": [],
        "proposed_updates": {
            "characters": [],
            "relationships": [],
            "address": [],
            "glossary": [],
        },
        "new_terms": [],
        "narrators": [],
        "summary_hint": "",
        "applied_updates": [],
    }


def _normalize_context(raw: Any, episode: int) -> dict[str, Any]:
    """Coerce a model response into the expected shape, dropping junk."""
    ctx = _empty_context(episode)
    if not isinstance(raw, dict):
        logger.warning("pass 1 returned non-object; ignoring segment")
        return ctx

    ctx["characters_present"] = _str_list(raw.get("characters_present"))
    ctx["new_terms"] = _str_list(raw.get("new_terms"))
    ctx["narrators"] = _str_list(raw.get("narrators"))
    ctx["summary_hint"] = str(raw.get("summary_hint") or "").strip()

    for pair in _dict_list(raw.get("speaker_pairs")):
        if pair.get("from") and pair.get("to"):
            ctx["speaker_pairs"].append(
                {
                    "from": str(pair["from"]),
                    "to": str(pair["to"]),
                    "register": str(pair.get("register", "unknown")),
                }
            )

    updates = raw.get("proposed_updates")
    if isinstance(updates, dict):
        ctx["proposed_updates"]["characters"] = _dict_list(updates.get("characters"))
        ctx["proposed_updates"]["relationships"] = _dict_list(updates.get("relationships"))
        ctx["proposed_updates"]["address"] = _dict_list(updates.get("address"))
        ctx["proposed_updates"]["glossary"] = _dict_list(updates.get("glossary"))
    return ctx


def _merge_into(target: dict[str, Any], seg: dict[str, Any]) -> None:
    _extend_unique(target["characters_present"], seg["characters_present"])
    _extend_unique(target["new_terms"], seg["new_terms"])
    _extend_unique(target["narrators"], seg["narrators"])
    if seg["summary_hint"]:
        target["summary_hint"] = (
            f"{target['summary_hint']} {seg['summary_hint']}".strip()
        )
    _merge_pairs(target["speaker_pairs"], seg["speaker_pairs"], ("from", "to"))
    tu, su = target["proposed_updates"], seg["proposed_updates"]
    _merge_pairs(tu["characters"], su["characters"], ("name",))
    _merge_pairs(tu["relationships"], su["relationships"], ("from", "to"))
    _merge_pairs(tu["address"], su["address"], ("from", "to"))
    _merge_pairs(tu["glossary"], su["glossary"], ("source",))


def _merge_pairs(
    target: list[dict[str, Any]], incoming: list[dict[str, Any]], keys: tuple[str, ...]
) -> None:
    """Append rows, overwriting an existing row with the same key tuple."""
    index = {tuple(row.get(k) for k in keys): row for row in target}
    for row in incoming:
        key = tuple(row.get(k) for k in keys)
        if None in key:
            continue
        if key in index:
            index[key].update(row)
        else:
            target.append(row)
            index[key] = row


def _extend_unique(target: list[str], incoming: list[str]) -> None:
    seen = set(target)
    for item in incoming:
        if item not in seen:
            target.append(item)
            seen.add(item)


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]
