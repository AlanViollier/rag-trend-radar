"""What a search hands on: hits and their stages, and the answers written from them."""

from datetime import date
from typing import Literal

from rag_trend_radar.records.base import Record


class SearchStage(Record):
    """One stage of a search and how long it took: `items` is what it handed on (candidates found,
    candidates reranked)."""

    name: Literal["embed", "search", "rerank"]
    seconds: float
    items: int


class SearchHit(Record):
    """One chunk returned by a search, best first."""

    chunk_id: int
    document_id: int
    text: str
    score: float
    context: str = ""  # what a short reply answered; `text` is only the reply's own words

    @property
    def full_text(self) -> str:
        """What the chunk was indexed and embedded from, and what a reranker should read."""
        return f"{self.text}\n\n{self.context}" if self.context else self.text


class Citation(Record):
    """A passage of a retrieved chunk that an answer relies on."""

    document_id: int
    cited_text: str


class Answer(Record):
    """A generated answer, its citations and what it cost."""

    question: str
    as_of: date
    text: str
    citations: tuple[Citation, ...]
    cost_usd: float
