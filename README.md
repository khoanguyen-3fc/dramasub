# dramasub

Context-aware drama subtitle translation with local LLMs via Ollama. Tracks
characters, relationships, and address terms per show for consistent
translations.

The primary use case is K-Drama (Korean → Vietnamese), but **languages are
configuration, not assumptions** — source and target are set per project.

## How it works

Every show is a **project**: a directory that accumulates context across
episodes (a series bible, a directed address table, a glossary, per-episode
summaries, and cached web context). Translation is **two-pass**:

1. **Pass 1** reads the episode and extracts structured context — who appears,
   who speaks to whom, relationship/speech-level changes, and new terms — and
   proposes bible updates (auto-applied, logged so you can revert).
2. **Pass 2** translates chunk by chunk, using a sliding window of already
   translated lines plus a bible excerpt filtered to the characters in each
   chunk.

The LLM never touches timestamps or file structure: subtitles are parsed and
reassembled programmatically, and every write is self-checked (cue count and
every timestamp must match the source, or the write aborts).

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with the model pulled:
  ```bash
  ollama pull qwen3.6:latest
  ```
- (Optional) a TMDB API key for online context, via `TMDB_API_KEY`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m dramasub.cli --help
```

## Usage

```bash
# 1. Create a project (defaults: source ko, target vi, model qwen3.6:latest)
python -m dramasub.cli init ./my-show --title "My Drama" --tmdb-id 123456

# 2. (Optional) fetch online context into the cache — see "Online context" below
export TMDB_API_KEY=...
python -m dramasub.cli context ./my-show --season 1 \
    --wiki-source "<source-edition article title>" --wiki-target "<target-edition title>"

# 3. Translate an episode (two-pass). Use -v to see per-chunk progress.
python -m dramasub.cli -v translate ./my-show --episode 1 path/to/E01.ko.srt

#    ...or a fast one-pass "direct" translate (no pass 1, TMDB context only):
python -m dramasub.cli -v translate ./my-show --episode 1 path/to/E01.ko.srt --direct

# 4. Inspect the accumulated bible (hand-edit the YAML to correct it)
python -m dramasub.cli bible ./my-show

# 5. Re-run quality checks on a translated episode
python -m dramasub.cli qc ./my-show --episode 1
```

Output lands in `my-show/episodes/e01/`:

```
source.srt      original subtitle (copied in, never modified)
context.yaml    pass-1 structured context
output.srt      translated subtitle (same timing as source)
summary.txt     recap that feeds the next episode's context
```

## Configuration

`project.yaml` holds per-project settings — languages, `model`, `num_ctx`,
`keep_alive`, `ollama_host` (the Ollama server URL, default
`http://localhost:11434`), `honorific_policy` (`translate` vs `keep_romanized`),
`loanword_policy` (`keep_english` vs `localize`), `romanization` (`media` vs
`revised`), chunk sizes, and per-pass temperatures. `bible.yaml` holds
characters (each with a **frozen** `target` name rendering so names never drift
between episodes), relationships, a **directed** address table (the
target-language terms a speaker uses for themself and the listener), and a
glossary. Both are human-editable — the tool only appends and updates, never
silently deletes.

Quality features drawn from professional subtitling practice: character names
are frozen on first sight and reused verbatim; pass 1 flags narrators so
commentary is rendered in the third person; and lines over the ~42-char reading
limit are re-requested more concisely rather than truncated.

## Online context (TMDB)

Online context is optional but improves quality a lot (see below). TMDB is the
primary source; supply a credential via the `TMDB_API_KEY` environment variable
(a local `.env` you `source`, or your shell) or `tmdb_api_key` in `project.yaml`.

**Which credential to use.** TMDB's API settings page shows two. Use the
**API Read Access Token** (v4 auth) — the long token beginning `eyJ…`:

```bash
export TMDB_API_KEY="eyJhbGciOiJIUzI1NiJ9...."
```

(The legacy **API Key** — the ~32-char hex string on the same page — also works;
dramasub auto-detects which one you provided.)

Fetched data (series overview, cast, episode synopses, in both source and
target languages) is cached under `<project>/cache/` and reused fully offline;
re-fetch only with `context --refresh`.

## Translation modes: what context buys you

Three ways to run the same episode, from cheapest to richest. Observed on a real
Korean→Vietnamese office drama (examples anonymized with the placeholder name
`홍길동` / *Hong Gil-dong*):

| | One-pass `--direct` | Two-pass, no TMDB | Two-pass + TMDB |
|---|---|---|---|
| Pass-1 analysis / bible | ✗ | ✓ | ✓ |
| Online context | TMDB only | ✗ | ✓ |
| Model calls / episode | ~½× | 1× | 1× + fetch |
| Names | correct **while** in the prompt | **guessed**, e.g. `Hong Gildong` | **official**, `Hong Gil-dong` |
| Cross-episode consistency | none (nothing frozen) | frozen after first sight | frozen from official spelling |
| Register / address | weakest | good | good |
| Full names not in the subtitle | ✗ | ✗ | **recovered from cast** (e.g. a given-name-only line gets its surname) |

What we actually saw:

- **No context** guesses romanizations ad hoc (`Hong Gildong`), occasionally
  mistranslates a job-title term literally (a rank word rendered as
  "responsibility"), and sometimes picks an odd register (an archaic "you").
- **+ TMDB** pins names to the official cast spellings, recovers a surname the
  subtitle never says (the cast list supplies it), fixes the title term, and
  reads more naturally.
- **`--direct`** is the cheapest and still gets names right *while TMDB sits in
  the prompt* — but with no bible it freezes nothing, so consistency isn't
  guaranteed across chunks or episodes, lines run longer, and subtext/register
  are weaker. Good for a fast draft; two-pass + TMDB is best for a finished sub.

(`--direct` is a lightweight one-pass path; the dedicated small-model real-time
mode remains on the backlog per [AGENTS.md](AGENTS.md).)

## Benchmark

Measured on a self-hosted Ollama server translating Korean→Vietnamese with the
default model `qwen3.6:latest` (a 35B-A3B MoE), `num_ctx=16384`, and TMDB
context cached:

| Hardware | |
|---|---|
| OS | Ubuntu 26.04 x86_64 |
| CPU | Intel Core i3-12100F (4C/8T) |
| GPU | NVIDIA GeForce RTX 3060 (12 GB) |
| RAM | 32 GiB |

| Mode | Throughput | Est. full episode (~900 cues) |
|---|---|---|
| Two-pass (pass 1 + pass 2) | ~12 cues/min | ~75 min |
| One-pass `--direct` | ~28 cues/min | ~33 min |

Wall-clock with the model already resident (`keep_alive`); the first call also
pays a one-time load. The 35B MoE only partially fits the 3060's 12 GB VRAM and
spills to CPU/RAM, so a larger card (or a smaller model) will be faster.
Two-pass does roughly 2× the model calls of direct, which the timings reflect.

## Design notes

The project follows [AGENTS.md](AGENTS.md): `dramasub.core` is a pure, UI-free
library (no prints, typed errors, stdlib logging); `dramasub.cli` is the only
place that prints. Correctness is enforced by runtime self-checks rather than a
test suite. Dependencies are limited to `pysubs2`, `requests`, and `PyYAML`.

## License

Released under the [MIT License](LICENSE).
