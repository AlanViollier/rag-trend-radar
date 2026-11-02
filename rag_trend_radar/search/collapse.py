"""Keep one of each near-identical chunk: one article pasted into five digests isn't five results."""

from collections.abc import Iterable

from rag_trend_radar.ingest.dedup import numbers, words
from rag_trend_radar.schema import SearchHit

## Share of the shorter text's words the longer one must hold. Read on 2,985 pairs of real results
## (2026-11-02): from .9 down it starts taking different mail for copies, wire reports for other
## amounts, meters, one trip's ticket for another's.
SHARED = 0.9
## Under this many distinct words a few shared ones say nothing, so only equal text counts.
MIN_WORDS = 15


def collapse(hits: Iterable[SearchHit], top: int) -> list[SearchHit]:
    """The first `top` hits of a best-first list that repeat none before them."""
    kept: list[tuple[SearchHit, str, set[str], set[str]]] = []
    for hit in hits:
        ## The chunk's lead differs between copies: only what follows it is compared.
        text = hit.text.partition("\n\n")[2] or hit.text
        new = (text.strip(), words(text), numbers(text))
        if not any(_same(new, seen[1:]) for seen in kept):
            kept.append((hit, *new))
        if len(kept) == top:
            break
    return [seen[0] for seen in kept]


def _same(a: tuple[str, set[str], set[str]], b: tuple[str, set[str], set[str]]) -> bool:
    text_a, words_a, numbers_a = a
    text_b, words_b, numbers_b = b
    shorter = min(len(words_a), len(words_b))
    if shorter < MIN_WORDS:
        return text_a == text_b
    ## A report with one figure changed is a different report: every number must be held.
    return len(words_a & words_b) / shorter >= SHARED and (
        numbers_a <= numbers_b or numbers_b <= numbers_a
    )
