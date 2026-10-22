"""Question plus retrieved chunks to Claude, answer back with native citations."""

from collections.abc import Sequence
from datetime import date

from rag_trend_radar.schema import Answer, SearchHit


def answer(question: str, hits: Sequence[SearchHit], as_of: date, model: str) -> Answer:
    """Send the hits as search_result blocks so every claim cites the chunk it came from."""
    raise NotImplementedError
