# AGENTS.md

Guidance for AI coding agents working on this repository.

## Project overview

**dramasub** (working name) is a local-first tool that translates drama subtitles
using local LLMs served by Ollama. The primary use case is K-Drama, Korean →
Vietnamese, but **languages are configuration, not assumptions**: source and
target language are set per project (defaults: `ko` → `vi`). Never hardcode
language-specific logic outside the prompt/config layer.

Core ideas:

1. **Every show is a project.** A project is a directory that accumulates context
   across episodes: series bible, character/address-term table, glossary,
   per-episode summaries, fetched web context, and translated outputs.
2. **Two-pass translation.** Pass 1 reads the episode subtitle and extracts
   structured context (who appears, who speaks to whom, relationship changes,
   new terms). Pass 2 translates chunk by chunk using that context plus the
   project bible.
3. **The LLM never touches timestamps or file structure.** Subtitle files are
   parsed and reassembled programmatically. The model only ever sees and returns
   numbered dialogue lines.

## Scope

**In scope now (v1):** the full-context two-pass pipeline, project management,
online context fetching, CLI.

**Backlog — do NOT implement unless a task explicitly asks:**

- **Real-time direct-translate mode.** One-pass, low-latency translation using a
  small model (~4–9B, e.g. `qwen3.5:latest` or `gemma4:e4b`) so it runs on
  modest hardware. This is the only place small models are used. Keep `core`
  design compatible (chunker and llm client should not assume two passes), but
  build nothing for it yet.
- **GUI.** A thin desktop shell (likely PySide6) over `dramasub.core`. This is
  why `core` must stay UI-free, but no GUI code yet.

## Tech stack — keep it simple

- Python 3.11+, plain `venv` + `pip` + `requirements.txt`. No uv, poetry,
  hatch, or lockfile tooling.
- Prefer the standard library. Third-party dependencies are limited to:
  - `pysubs2` — subtitle parsing (SRT and ASS/SSA, preserves timing/styling)
  - `requests` — HTTP (Ollama API, TMDB, Wikipedia)
  - `PyYAML` — project/bible files
- LLM runtime: Ollama via its local REST API (`http://localhost:11434`),
  called directly with `requests`. No LangChain or similar frameworks. No
  cloud LLM providers.
- CLI: stdlib `argparse`. No typer/click/rich.
- No test framework. Correctness is enforced by **runtime self-checks** built
  into the pipeline (see "Runtime self-checks" below) instead of a test suite.
  Do not add pytest/unittest scaffolding.
- Validation: plain dataclasses + explicit validation functions. No pydantic.
- Formatting: follow PEP 8; no enforced tooling in the repo.

## Repository layout

```
dramasub/
  core/               # Pure library. No UI imports, no print(). Raise or return.
    project.py        # Project (show) lifecycle: create, load, save
    bible.py          # Series bible + address table + glossary
    context_tmdb.py   # TMDB API client (localized: source + target language)
    context_wiki.py   # Wikipedia REST API client (source + target editions)
    subtitle.py       # Parse/reassemble SRT/ASS via pysubs2; cue indexing
    chunker.py        # Chunking + sliding window assembly
    llm.py            # Ollama client: generate, JSON output, retries, keep_alive
    pass1.py          # Episode context extraction
    pass2.py          # Chunk translation
    qc.py             # Optional QC pass (glossary/consistency check)
    prompts/          # One file per prompt template, with a version comment
  cli.py              # argparse entry point; the only place that prints
requirements.txt
```

**Dependency rule:** `cli.py` imports `core`; `core` never imports `cli`.
Everything in `core` must be usable headlessly — this is what makes the
backlogged GUI and real-time mode cheap later.

## Project (show) directory format

A project lives in a user-chosen directory:

```
<project>/
  project.yaml        # title, TMDB id, source/target language, model config,
                      # honorific policy
  bible.yaml          # characters, relationships, address table, glossary
  cache/              # raw fetched web context (JSON, keyed by source)
  episodes/
    e01/
      source.srt      # original subtitle (copied in, never modified)
      context.yaml    # pass-1 output: structured episode context
      output.srt      # translated subtitle
      summary.txt     # post-episode summary, feeds the next episode's context
```

- All YAML files must be human-editable; users will hand-correct the bible and
  address table. Preserve unknown keys on round-trip; don't reformat beyond
  what a change requires.
- `bible.yaml` is append/update only from code; never silently delete entries
  the model didn't mention this episode.

## Domain rules (important — this is where quality lives)

### Address terms / speech registers

The hardest problem in translating between Asian languages. The source language
may encode speech levels and address terms (Korean 반말/존댓말, 오빠, 선배…);
the target may require consistent per-pair pronoun choices (Vietnamese xưng hô:
anh/em/chị/tôi/cậu…). The bible stores a **directed** address table — A→B may
differ from B→A:

```yaml
address:
  - from: Ji-ho
    to: Min-seo
    self: anh        # target-language term the speaker uses for themself
    other: em        # term used for the listener
    since_episode: 5
    note: "dating from ep 5; used 'tôi/cô' in ep 1-4"
```

- The `self`/`other` values are free-form target-language strings — the schema
  is language-neutral even though the defaults document Vietnamese.
- Pass 2 prompts must include the address-table rows for every character pair
  present in the chunk.
- Pass 1 must flag relationship/speech-level changes (confessions, fights,
  first informal speech) as proposed bible updates — auto-applied but logged so
  the user can review and revert.
- The honorific policy (`keep_romanized` vs `translate`) is set per project in
  `project.yaml` and injected into every translation prompt.

### Subtitle handling

- Parse with `pysubs2`. Internally address cues by integer index.
- Send the model only `{index: text}` pairs; require JSON back mapping the same
  indices to translations. Validate: same key set, no empty values. On
  mismatch, retry the chunk up to 2 times with an error hint; on final
  failure, mark cues untranslated and continue — never crash mid-episode;
  report failures at the end.
- Preserve ASS styling tags and multi-line cues. Strip inline tags before
  sending to the model when possible and re-apply after; otherwise instruct
  the model to preserve `{...}` and `\N` verbatim and validate that it did.
- Soft limit ≈ 42 characters per rendered output line (prompt constraint plus
  QC warning — never hard-truncate).

### Chunking and context windows

- Chunk size: 10–15 cues.
- Sliding window: include the previous ~8 cues **with their finalized
  translations** plus ~4 source-only lookahead cues (verb-final languages often
  resolve meaning in the next cue).
- Filter bible excerpts to characters present in the chunk; never dump the
  whole bible into a prompt.

### Runtime self-checks (replaces a test suite)

The pipeline must verify its own output on every run:

- After reassembling `output.srt`, assert the cue count and **every timestamp**
  match `source.srt` exactly; abort the write and report if not.
- Per-chunk validation as described above (index sets match, no empty values,
  retry then skip-and-report).
- End each `translate` run with a summary: cues translated, cues failed (with
  indices), QC warnings (over-length lines, glossary mismatches).
- These checks are mandatory in every code path that writes subtitle output —
  including future backlog features.

### LLM usage

- Default model for the full pipeline: **`qwen3.6:latest`** (35B-A3B MoE — near
  large-model quality with ~3B active parameters). Model name is configurable
  per project in `project.yaml`; never hardcode it outside config defaults.
- Small models are reserved for the backlogged real-time mode only.
- Always set `num_ctx` explicitly in every Ollama request (default 16384,
  configurable). Ollama's implicit default is tiny and silently truncates.
- Qwen3.6 thinks by default: disable thinking where supported; in all cases,
  strip `<think>...</think>` before parsing. Output parsing must tolerate
  leading/trailing junk and markdown fences around JSON.
- Set `keep_alive` (e.g. `30m`) so the model stays loaded between chunks.
- Use Ollama's `format: json` where it helps; still validate the result.
- Temperature: low (≈0.3) for pass 2; moderate (≈0.7) for pass 1 and summaries.

### Online context

- TMDB is the primary source (official API; supports localized queries). Fetch
  series overview, episode synopses, and cast/character list in **both** the
  source and target languages when available. API key from env var
  `TMDB_API_KEY` or project config; never commit keys.
- Wikipedia REST API: source-language edition for accurate native names and
  relationships; target-language edition for community-standard name
  renderings.
- All fetched data is cached in `<project>/cache/` and refreshed only on
  explicit user request. The tool must work fully offline once a project's
  cache and bible exist.
- MyDramaList/AsianWiki have no official APIs — out of scope; do not add
  scrapers without an explicit task.

## Coding conventions

- Type hints on public functions in `core`.
- No global state; pass a `Project` object explicitly.
- `core` raises typed exceptions (`DramasubError` subclasses); `cli.py`
  catches and prints them.
- Logging via the stdlib `logging` module in `core`; printing only in `cli.py`.
- Every prompt template lives in `core/prompts/` as a separate file with a
  version comment — no inline prompt strings scattered through logic.
- LLM calls go through one interface in `llm.py` so tests can inject canned
  responses. No test may require a running Ollama instance.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m dramasub.cli --help           # CLI entry point
```

Planned CLI surface (implement in this order):

```bash
python -m dramasub.cli init <dir> --title "..." [--tmdb-id N] [--source ko] [--target vi]
python -m dramasub.cli bible <project>                    # show bible; edit by opening the YAML
python -m dramasub.cli translate <project> --episode 1 <file.srt>
python -m dramasub.cli qc <project> --episode 1
```

## Definition of done for a task

- Any code path that writes subtitle output includes the runtime self-checks
  (timestamp/count verification, chunk validation, end-of-run summary).
- LLM-facing changes update the corresponding validation code in the same
  change.
- If a change affects prompt content, note it in the prompt file's version
  comment.
- User-editable file formats (`project.yaml`, `bible.yaml`) stay backward
  compatible, or a migration is included.
