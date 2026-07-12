"""Chunking and sliding-window assembly.

Pass 2 translates a chunk at a time. Each chunk carries a sliding window so
the model has continuity without seeing the whole episode:

* **cues** — the 10–15 cues to translate now.
* **prev** — the preceding cues. Pass 2 renders these *with their finalized
  translations* (looked up from its running results), giving the model the
  target-language context it must stay consistent with.
* **lookahead** — the following cues, **source only**. Verb-final languages
  like Korean often resolve meaning in the next line, so the model gets to
  peek ahead before committing a translation.

The window is expressed as references to :class:`~dramasub.core.subtitle.Cue`
objects; the *translations* themselves live in pass 2's accumulator and are
joined in at prompt-render time. This module is pure structure — it knows
nothing about the bible, prompts, or the LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from dramasub.core.errors import ValidationError
from dramasub.core.subtitle import Cue

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A unit of translation plus its sliding-window context."""

    number: int  # 1-based, for logging/progress
    cues: list[Cue]
    prev: list[Cue] = field(default_factory=list)
    lookahead: list[Cue] = field(default_factory=list)

    @property
    def indices(self) -> list[int]:
        return [c.index for c in self.cues]

    def source_map(self) -> dict[int, str]:
        """The ``{index: body}`` payload the model must translate."""
        return {c.index: c.body for c in self.cues}


def plan(
    cues: Sequence[Cue],
    *,
    size: int,
    prev_window: int,
    lookahead: int,
) -> list[Chunk]:
    """Split *cues* (the ordered translatable cues) into windowed chunks.

    *cues* must already be filtered to translatable cues in document order —
    typically ``[doc.cue(i) for i in doc.translatable_indices()]``. A tiny
    trailing chunk is merged into its predecessor so no chunk is stranded
    with one or two cues.
    """
    if size < 1:
        raise ValidationError(f"chunk size must be >= 1, got {size}")
    if prev_window < 0 or lookahead < 0:
        raise ValidationError("prev_window and lookahead must be >= 0")
    if not cues:
        return []

    groups = _group(list(cues), size)

    chunks: list[Chunk] = []
    cursor = 0
    for number, group in enumerate(groups, start=1):
        start = cursor
        end = cursor + len(group)
        prev = list(cues[max(0, start - prev_window):start])
        ahead = list(cues[end:end + lookahead])
        chunks.append(Chunk(number=number, cues=group, prev=prev, lookahead=ahead))
        cursor = end

    logger.info(
        "planned %d chunk(s) over %d cues (size=%d, prev=%d, lookahead=%d)",
        len(chunks),
        len(cues),
        size,
        prev_window,
        lookahead,
    )
    return chunks


def _group(cues: list[Cue], size: int) -> list[list[Cue]]:
    groups = [cues[i:i + size] for i in range(0, len(cues), size)]
    # Avoid a stranded final chunk: fold a small remainder into the one before
    # it, keeping chunk sizes in a sensible band.
    min_tail = max(2, size // 3)
    if len(groups) >= 2 and len(groups[-1]) < min_tail:
        tail = groups.pop()
        groups[-1].extend(tail)
    return groups
