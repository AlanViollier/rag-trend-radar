"""Keep press and bulk mail from filling the results, so what people wrote inside gets seen."""

from collections.abc import Iterable

from rag_trend_radar.ingest.kinds import Source, source
from rag_trend_radar.schema import SearchHit

## An investigator's top 10 was 91% press on story questions (109 of 120, 2026-11-02), with press
## about 6% of the chunks. At most this many of a list are press or bulk mail, unless nothing
## else is left.
RESULTS = 10
MAX_PRESS = 3
MAX_BULK = 1
## A rerank score under this is noise ("UNBELIEVABLE", "FYI"): such a chunk never takes the place
## of a press article.
MIN_SCORE = 0.05


def mix(hits: Iterable[SearchHit], top: int) -> list[SearchHit]:
    """The best `top` of a best-first list under the caps, best first.

    What the caps hold back comes back only when nothing else is left, ahead of mail scoring
    under `MIN_SCORE`.
    """
    limits = {Source.MAIL: top, Source.PRESS: MAX_PRESS, Source.BULK: MAX_BULK}
    taken = dict.fromkeys(Source, 0)
    chosen: list[SearchHit] = []
    held: list[SearchHit] = []
    noise: list[SearchHit] = []
    for hit in hits:
        kind = source(hit.text)
        if kind is Source.MAIL and hit.score < MIN_SCORE:
            noise.append(hit)
        elif taken[kind] < limits[kind]:
            taken[kind] += 1
            chosen.append(hit)
        else:
            held.append(hit)
    chosen += (held + noise)[: max(top - len(chosen), 0)]
    return sorted(chosen[:top], key=lambda hit: -hit.score)
