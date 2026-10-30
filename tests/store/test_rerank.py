from collections.abc import Sequence

import psycopg
import pytest

from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search import rerank as module
from rag_trend_radar.search.rerank import rerank


class _Counting:
    """Scores a pair by its chunk's length and counts every pair it was asked about."""

    def __init__(self) -> None:
        self.pairs = 0

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs += len(pairs)
        return [float(len(text)) for _, text in pairs]


def _hits(*texts: str) -> list[SearchHit]:
    return [SearchHit(chunk_id=i, document_id=i, text=t, score=0.0) for i, t in enumerate(texts)]


@pytest.fixture
def encoder(store: None, monkeypatch: pytest.MonkeyPatch) -> _Counting:
    with db.connect() as conn:
        conn.execute("TRUNCATE cache.rerank_scores")
    counting = _Counting()
    monkeypatch.setattr(module, "_load", lambda model, revision: counting)
    return counting


def test_a_pair_scored_once_is_read_back_not_scored_again(encoder: _Counting) -> None:
    first = rerank("q", _hits("bb", "a", "ccc"), "m", top=3)
    second = rerank("q", _hits("bb", "a", "ccc", "dddd"), "m", top=3)
    assert [h.text for h in first] == ["ccc", "bb", "a"]
    assert [h.text for h in second] == ["dddd", "ccc", "bb"]
    assert encoder.pairs == 4


def test_another_question_scores_the_same_chunks_afresh(encoder: _Counting) -> None:
    rerank("q", _hits("a", "bb"), "m", top=2)
    rerank("another q", _hits("a", "bb"), "m", top=2)
    assert encoder.pairs == 4


@pytest.mark.usefixtures("store")
def test_readers_cant_touch_the_score_cache() -> None:
    with db.connect("investigator") as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("INSERT INTO cache.rerank_scores VALUES ('m', 'r', 'q', 't', 1e9)")
