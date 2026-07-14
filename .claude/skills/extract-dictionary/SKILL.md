---
name: extract-dictionary
description: >-
  Build a universal source→target subtitle dictionary from parallel/reference
  subtitle pairs, then prune it with a target Ollama model so only entries that
  model gets wrong on its own survive. Use to create or extend a dramasub default
  dictionary for a new language pair (e.g. ko→vi) or genre. Inputs: source
  language A, target language B, reference subtitle pairs, and the model X the
  dictionary will guide.
---

# extract-dictionary

Produce a lean, high-signal `A→B` glossary from human reference subtitles, keeping
**only** the terms model **X** mistranslates unaided. Everything model X already
handles is dropped — the dictionary is a guide, so redundant entries are noise.

Helper scripts live beside this file; run them with the project venv
(`.venv/bin/python`). They depend only on `pysubs2` / `requests` / `PyYAML`.

## Inputs (parse from the user's request; ask via AskUserQuestion only if missing)
- **A** — source language: name + code (e.g. Korean / `ko`).
- **B** — target language: name + code (e.g. Vietnamese / `vi`).
- **pairs** — for each work, an A subtitle and its time-aligned B reference. Give a
  directory + suffixes (`*.kor.srt` + `*.vie.srt`) or an explicit `A:B,…` list.
  More *distinct works* = stronger universality signal; several episodes of one
  show mostly re-confirm the same show, so prefer breadth across works.
- **X** — the Ollama model the dictionary will guide / be pruned against
  (e.g. `gemma4:latest`). Host comes from `$OLLAMA_HOST` / a local `.env`.
- **out** — output path (default `dictionary.<a>-<b>.yaml`).
- **known** (optional) — an existing dictionary to *extend*; its sources are
  excluded from new proposals.

## Workflow

### 1 — Align
```
.venv/bin/python .claude/skills/extract-dictionary/align_pairs.py \
  --dir SUBS --a-suffix .kor.srt --b-suffix .vie.srt [--known DICT.yaml] --out aligned.json
```
Output: `{known_sources, files: {work: [{a, b}]}}` — each A cue paired with the
time-overlapping B reference text.

### 2 — Mine (parallel bilingual sub-agents, one per work)
Write each work's `{known_sources, pairs}` to its own file and dispatch a
sub-agent per work (run them concurrently). Prompt template — fill {A}/{B}/{path}:

> You are a bilingual {A}→{B} lexicographer building a UNIVERSAL, work-agnostic
> subtitle dictionary. Read the aligned corpus at {path} (JSON
> `{known_sources, pairs:[{a,b}]}`; the B side is a human reference that may drift
> or be relayed through a third language — prefer natural, source-faithful {B}).
> Propose NEW entries: recurring, GENERIC {A} vocabulary / idioms / discourse
> markers that would appear across MANY works. STRICT rules: universal only;
> EXCLUDE character/person/place names, brand/product/company coinages, and
> plot-specific jargon (when unsure, EXCLUDE); EXCLUDE anything in known_sources
> and obvious inflections; prefer terms seen more than once; give a natural
> base-form {B} rendering. Return ONLY JSON:
> `{"terms":[{"source","target","note","freq","why_universal"}], "guide_rules":[{"rule","example"}]}`

**Consolidate:** dedupe by `source`; the strong signal is a term proposed for
**different works**. Keep cross-work terms + clearly-universal single-work ones;
drop anything that slipped through as work-specific. This is the candidate set.

### 3 — Prune with model X (blind translation)
Write the candidate sources to `cand.json` (a JSON list), then:
```
.venv/bin/python .claude/skills/extract-dictionary/blind_translate.py \
  --terms cand.json --model X --source-name {A} --target-name {B} --out blind.json
```
This is model X translating each term **without** the dictionary — the direct test
of whether X needs the hint.

### 4 — Judge (parallel bilingual sub-agents)
Split `blind.json` (each `{source, target(=candidate rendering), blind}`) across a
few judge sub-agents. Decision per entry:
- **REMOVE** if `blind` already conveys `target` with acceptable meaning/register
  (minor synonym differences still count as fine).
- **KEEP** only on a real divergence: wrong meaning; wrong register/pronoun/rank;
  a calqued or mistranslated idiom; a false friend; left in another language; an
  English loanword the dict localizes (or vice-versa); or a culture-specific gap.
- **Bias toward REMOVE** — the goal is to trim redundancy.

Judge prompt: give the file path, the criteria above, and require
`{"remove":[…], "keep":[…], "notes":[…]}` with every source in exactly one list.

**Force-keep** any term you have separately observed failing in real end-to-end
translation runs even if its isolated `blind` looks fine (isolated ≠ in-context).

### 5 — Write
```
.venv/bin/python .claude/skills/extract-dictionary/write_dictionary.py \
  --entries kept.json --pair <a>-<b> --version N \
  --header "how it was built — anonymized, NO work/show names" --out <out>
```
`kept.json` = `[{source, target, note}]` for the KEEP set (+ force-keeps). The
provenance header must use generic genre descriptions, never a work title.

## Guardrails
- **Never** put a work/show/character/place/brand name in the dictionary (entries,
  notes, or comments). Examples use placeholder names only.
- The dictionary is merged *under* a project's bible glossary and is **not**
  QC-enforced — a lean set of only-what-X-gets-wrong is the target, not coverage.
- Scale agents to the corpus (one miner per work; judges ~100 terms each).
- A B-reference relayed through a third language drifts — judge and mine against
  the A source, not the reference wording.
