"""Keep one of each near-identical chunk: one article pasted into five digests isn't five results."""

from collections.abc import Iterable
from typing import NamedTuple

from rag_trend_radar.ingest.dedup import numbers, words
from rag_trend_radar.records import SearchHit

# Share of a chunk's words a better one must hold for it to drop. Read on 2,985 pairs of real
# results (2026-11-02): below .9 it takes different mail for copies, wire reports for other
# amounts, meters, one trip's ticket for another's.
SHARED = 0.9
# Under this many distinct words a few shared ones say nothing, so only equal text counts.
MIN_WORDS = 15


def collapse(hits: Iterable[SearchHit], top: int) -> list[SearchHit]:
    """The first `top` hits of a best-first list, without those a better hit already says."""
    kept: list[tuple[SearchHit, _Text]] = []
    for hit in hits:
        # The chunk's lead differs between copies: only what follows it is compared.
        text = hit.text.partition("\n\n")[2] or hit.text
        new = _Text(text.strip(), words(text), numbers(text))
        if not any(_repeats(new, seen) for _, seen in kept):
            kept.append((hit, new))
        if len(kept) == top:
            break
    return [hit for hit, _ in kept]


class _Text(NamedTuple):
    text: str
    words: set[str]
    numbers: set[str]


def _repeats(new: _Text, better: _Text) -> bool:
    # One way only: a short chunk ranked higher must not take a longer one holding more facts
    # (it dropped the one email naming who to contact, 2026-11-02). A number it adds keeps it.
    if len(new.words) < MIN_WORDS:
        return new.text == better.text
    return (
        len(new.words & better.words) / len(new.words) >= SHARED and new.numbers <= better.numbers
    )
