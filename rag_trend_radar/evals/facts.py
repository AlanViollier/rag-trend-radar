"""Hard questions: facts that must appear in the results an answer is written from, checked by code."""

import re
from collections.abc import Sequence

from rag_trend_radar.ingest.kinds import Source, source
from rag_trend_radar.schema import FactQuestion, FactResult, SearchHit


def check(question: FactQuestion, hits: Sequence[SearchHit]) -> FactResult:
    """Which required facts `hits` hold, and whether any forbidden text came back.

    A fact is a case-insensitive pattern searched in each chunk as it reaches an answer, its lead
    and a short reply's quote included. With `needs_mail`, only chunks from people's own mail
    count: a press report of the fact won't do.
    """
    texts = [h.full_text for h in hits if not question.needs_mail or source(h.text) is Source.MAIL]
    missing = tuple(f for f in question.facts if not any(_found(f, text) for text in texts))
    leaked = tuple(f for f in question.forbid if any(_found(f, hit.full_text) for hit in hits))
    return FactResult(
        id=question.id,
        missing=missing,
        leaked=leaked,
        top_score=max((hit.score for hit in hits), default=0.0),
        mail=sum(source(hit.text) is Source.MAIL for hit in hits),
    )


def _found(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None
