# Models & quality

[← README](../README.md)

## Choosing a model

The default is **`gemma4:latest`** (8B). In a blind three-judge comparison — the
Korean source as reference, model identities hidden — it was ranked best of the
models tried, with the highest adequacy and fluency and a clear majority of
per-cue wins, *and* it is the smallest and fastest, fitting entirely in a 12 GB
GPU.

| Model | Size | Fits 12 GB VRAM | Speed | Quality (blind judges) |
|---|---|---|---|---|
| `gemma4:latest` (**default**) | 8B | ✓ fully on GPU | fastest | best — top adequacy + fluency |
| `qwen3.5:latest` | 9.7B | ✓ fully on GPU | fast | decent; some meaning reversals |
| `qwen3.6:latest` | 35B-A3B MoE | ✗ spills to CPU/RAM | slow | strong meaning, but slower, with occasional wrong register / hallucination |

Set the model with `init --model <name>` or the `model` key in `project.yaml`.

## What holds across every test

- **Direct from Korean beats an English pivot.** Scored against the *Korean* — not
  the Vietnamese reference, which was itself relayed through English — the local
  pipeline kept meaning the pivoted reference lost; in one episode the reference
  even followed a mis-aligned English line unrelated to the Korean.
- **Adequacy is near a human team; the gap is polish.** Against a third-party
  **human** sub over 5 episodes, the pipeline's meaning fidelity was near-even
  (~7.0 vs ~7.8 on a 1–10 scale); the human led mainly on fluency and register —
  exactly what the directed address table and glossary exist to close with a
  little curation.
- **The recurring errors are data-fixable.** A wrong pronoun for a relationship, a
  mistranslated job title, or a bit of untranslated slang is fixed once in the
  bible/glossary and inherited by every later cue and episode.

Method: judgments are by stronger models on Korean-source samples (the model
comparison was one episode, blind, three judges; the human-team comparison ran 5
episodes) — directional, not a leaderboard. Speed figures used an RTX 3060
(12 GB) / i3-12100F / 32 GiB on Ubuntu, `num_ctx=16384`, TMDB cached.

<details>
<summary>Historical note — qwen3.5 vs qwen3.6, before gemma4</summary>

The two Qwen models were compared first, over 5 episodes (240 cues) judged
against the Korean. `qwen3.6:latest` scored higher (adequacy 7.7 vs 5.8) but ran
~5× slower because its 35B weights spill off a 12 GB GPU (~12 vs ~58 cues/min,
two-pass), while `qwen3.5:latest` fit fully in VRAM. `qwen3.5` also showed more
meaning reversals and leftover untranslated English. Both are kept as options,
but `gemma4` now supersedes them as the default: faster than qwen3.5 and judged
higher-quality than qwen3.6.
</details>

## Limitations

- **Not at parity with a skilled human on polish.** A human sub reads more
  naturally and handles register (pronouns / speech levels) more reliably; out of
  the box the model sometimes defaults to a wrong or archaic pronoun until the
  address table is curated.
- **Register and culture-specific terms need review.** Job titles, honorifics, and
  slang can come out wrong on the first pass — the bible and glossary exist to fix
  this, but that is human work.
- **Quality is bounded by the local model.** An 8–35B local model won't match a
  frontier model's fluency; expect occasional mistranslations, code-switching (a
  stray English/Korean/Chinese word — caught and retried), or a hallucinated
  detail.
- **Bigger isn't always better or faster.** The largest model tried (~35B MoE)
  spills off a 12 GB GPU and runs several times slower without beating the 8B
  default on this task.
- **Wordplay and brand slang still need a human.** Puns and in-scene shorthand are
  not reliably reinvented.

## Translation modes: what context buys you

Three ways to run the same episode, cheapest to richest. Observed on a real
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
| Full names not in the subtitle | ✗ | ✗ | **recovered from cast** |

- **No context** guesses romanizations ad hoc (`Hong Gildong`), occasionally
  mistranslates a job-title term literally, and sometimes picks an odd register.
- **+ TMDB** pins names to the official cast spellings, recovers a surname the
  subtitle never says, fixes the title term, and reads more naturally.
- **`--direct`** is cheapest and still gets names right *while TMDB sits in the
  prompt* — but freezes nothing, so consistency isn't guaranteed across chunks or
  episodes. Good for a fast draft; two-pass + TMDB is best for a finished sub.

(`--direct` is a lightweight one-pass path; the dedicated small-model real-time
mode remains on the backlog per [AGENTS.md](../AGENTS.md).)
