# Guide & dictionary

[← README](../README.md)

dramasub ships two packaged defaults for **Korean → Vietnamese**, applied
automatically when a project's language pair matches:

- a **translation guide** — dense prose rules injected into the pass-2 prompt
  (pronoun-pair locks, honorific-ending → particle mappings, title handling,
  interjections-by-function, "sentence-opening 아니 is surprise, not the negation
  không", regional dialect flattens to neutral, etc.);
- a **base dictionary** — `source → target` term renderings.

The dictionary sits **under** the show's bible glossary (the bible always wins per
term) and is filtered per chunk. It **guides** generation but is **not** enforced
by QC — only the curated bible glossary is.

## How the dictionary is built

The goal is a **compact, high-signal** set, not broad coverage — a hint the model
doesn't need is noise. It's built in two stages:

1. **Mine.** Parallel reference subtitles (source + a human target sub) from
   several works are timestamp-aligned, and bilingual passes propose recurring,
   *work-agnostic* term/idiom candidates. Cross-work agreement is the universality
   signal; character/place/brand/product names and plot coinages are excluded.
2. **Prune with the target model.** Every candidate is blind-translated by the
   model the dictionary will guide (e.g. `gemma4`). An entry is **kept only if the
   model gets it wrong on its own** — a scrambled workplace rank, a literal kinship
   pronoun, a calqued idiom, a false friend, a flattened register/euphemism, or a
   culture-specific gap. Everything the model already handles is dropped.

Two deliberate exceptions to "prune what the model gets right":

- The **workplace-rank ladder** (사원 … 회장) is kept whole even where a rank is
  individually easy, because ranks disambiguate each other — without the
  neighbours the model mis-ranks an ambiguous one (e.g. 책임).
- A **context-dependent false friend** keeps a note rather than a hard rule
  (미팅 is usually a group blind date, but a work 미팅 is a business meeting).

The packaged ko→vi dictionary is therefore small on purpose. Show-specific names,
places, and product coinages never appear; examples use placeholder names only.

## Customization

Control both defaults per project in `project.yaml`:

```yaml
guide: default        # 'default' | 'none' | ./path/to/guide.txt
dictionary: default   # 'default' | 'none' | ./path/to/dictionary.yaml
```

`default` uses the packaged file for the project's language pair (if one exists),
`none` disables it, and a path loads your own. This is how another genre or
language pair brings its own.

Dictionary format:

```yaml
version: 1
terms:
  - source: 책임
    target: chủ nhiệm
    note: senior/principal rank; title-first ("김 책임님" -> "Chủ nhiệm Kim")
```

## Generating a dictionary for a new pair

The [`extract-dictionary`](../.claude/skills/extract-dictionary/SKILL.md) skill
automates the whole pipeline above for **any** language pair, reference-subtitle
set, and Ollama model:

```
align (timestamp-pair source ↔ reference)
  → mine (parallel bilingual passes; cross-work agreement)
  → prune with model X (blind-translate; keep only what X gets wrong)
  → judge → write dictionary.<a>-<b>.yaml
```

Its three helper scripts depend only on `pysubs2` / `requests` / `PyYAML`:
`align_pairs.py`, `blind_translate.py`, `write_dictionary.py`. See the skill's
`SKILL.md` for the playbook and guardrails (never emit work/show names; keep the
set lean).
