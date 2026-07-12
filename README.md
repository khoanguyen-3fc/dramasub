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
python -m dramasub.cli init ./my-show --title "See You at Work Tomorrow" --tmdb-id 123456

# 2. (Optional) fetch online context into the cache
export TMDB_API_KEY=...
python -m dramasub.cli context ./my-show --season 1 \
    --wiki-source "내일 봐요 사장님" --wiki-target "..."

# 3. Translate an episode (two-pass). Use -v to see per-chunk progress.
python -m dramasub.cli -v translate ./my-show --episode 1 path/to/E01.ko.srt

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
`keep_alive`, `honorific_policy` (`translate` vs `keep_romanized`), chunk sizes,
and per-pass temperatures. `bible.yaml` holds characters, relationships, a
**directed** address table (the target-language terms a speaker uses for
themself and the listener), and a glossary. Both are human-editable — the tool
only appends and updates, never silently deletes.

## Design notes

The project follows [AGENTS.md](AGENTS.md): `dramasub.core` is a pure, UI-free
library (no prints, typed errors, stdlib logging); `dramasub.cli` is the only
place that prints. Correctness is enforced by runtime self-checks rather than a
test suite. Dependencies are limited to `pysubs2`, `requests`, and `PyYAML`.
