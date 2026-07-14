# Configuration

[← README](../README.md)

## `project.yaml`

Per-project settings:

| Key | Meaning |
|---|---|
| `source_language` / `target_language` | ISO codes, e.g. `ko` / `vi` |
| `model` | Ollama model tag (default `gemma4:latest`) |
| `num_ctx`, `keep_alive` | Ollama context window and model residency |
| `ollama_host` | server URL (default `http://localhost:11434`) |
| `honorific_policy` | `translate` vs `keep_romanized` |
| `loanword_policy` | `keep_english` vs `localize` |
| `romanization` | `media` vs `revised` |
| `chunk` | pass-2 chunk `size`, `prev_window`, `lookahead` |
| `temperature` | per-pass temperatures (`pass1`, `pass2`, `summary`) |
| `retry_temperatures` | temperatures for pass-2 retries (hotter retries help the model escape a stubborn wrong output) |
| `guide` / `dictionary` | `default`, `none`, or a path — see [dictionary.md](dictionary.md) |

## `.env`

A `.env` file in the working directory is loaded automatically (existing
environment always wins), so secrets and machine-specific settings live outside
`project.yaml`. Keep it gitignored.

```dotenv
OLLAMA_HOST=http://10.0.0.20:11434
TMDB_API_KEY=eyJhbGci...
```

Recognized keys: `TMDB_API_KEY`, and `OLLAMA_HOST` (overrides `ollama_host`; a
bare `host:port` gets an `http://` prefix).

## `bible.yaml`

Holds characters (each with a **frozen** `target` name rendering so names never
drift between episodes), relationships, a **directed** address table (the
target-language terms a speaker uses for themselves and for the listener), and a
glossary. Human-editable — the tool only appends and updates, never silently
deletes.

Quality features drawn from professional subtitling practice: character names are
frozen on first sight and reused verbatim; pass 1 flags narrators so commentary is
rendered in the third person; and lines over the ~42-char reading limit are
re-wrapped (or re-requested more concisely) rather than truncated.

## Online context (TMDB)

Online context is optional but improves quality a lot (see
[models.md → translation modes](models.md#translation-modes-what-context-buys-you)).
TMDB is the primary source; supply a credential via the `TMDB_API_KEY` environment
variable (in `.env` or your shell) or `tmdb_api_key` in `project.yaml`.

**Which credential.** TMDB's API settings page shows two. Use the **API Read
Access Token** (v4 auth) — the long token beginning `eyJ…`:

```bash
export TMDB_API_KEY="eyJhbGciOiJIUzI1NiJ9...."
```

(The legacy **API Key** — the ~32-char hex string on the same page — also works;
dramasub auto-detects which one you provided.)

Fetched data (series overview, cast, episode synopses, in both source and target
languages) is cached under `<project>/cache/` and reused fully offline; re-fetch
only with `context --refresh`. Wikipedia can supplement it via
`context --wiki-source "<title>" --wiki-target "<title>"`.
