from collections.abc import Sequence

import psycopg
import pytest

from rag_trend_radar.db import connect as db
from rag_trend_radar.records import SearchHit
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


def _rerank(query: str, hits: list[SearchHit], top: int) -> list[SearchHit]:
    # As a reader, the way every search reranks.
    with db.connect("investigator") as conn:
        return rerank(conn, query, hits, "m", top=top)


def test_a_pair_scored_once_is_read_back_not_scored_again(encoder: _Counting) -> None:
    first = _rerank("q", _hits("bb", "a", "ccc"), top=3)
    second = _rerank("q", _hits("bb", "a", "ccc", "dddd"), top=3)
    assert [h.text for h in first] == ["ccc", "bb", "a"]
    assert [h.text for h in second] == ["dddd", "ccc", "bb"]
    assert encoder.pairs == 4


def test_another_question_scores_the_same_chunks_afresh(encoder: _Counting) -> None:
    _rerank("q", _hits("a", "bb"), top=2)
    _rerank("another q", _hits("a", "bb"), top=2)
    assert encoder.pairs == 4


def test_the_reranker_reads_a_short_replys_quote(encoder: _Counting) -> None:
    short = SearchHit(chunk_id=1, document_id=1, text="ok", context="a long quote", score=0.0)
    longer = SearchHit(chunk_id=2, document_id=2, text="a longer body", score=0.0)
    assert [h.chunk_id for h in _rerank("q", [longer, short], top=2)] == [1, 2]


@pytest.mark.usefixtures("store")
@pytest.mark.parametrize(
    "change",
    [
        "UPDATE cache.rerank_scores SET score = 1e9",
        "DELETE FROM cache.rerank_scores",
        "INSERT INTO cache.rerank_scores VALUES ('m', 'r', 'q', 't', 2) "
        "ON CONFLICT (model, revision, query_hash, text_hash) DO UPDATE SET score = 1e9",
    ],
)
def test_a_reader_adds_scores_but_never_changes_one(change: str) -> None:
    with db.connect() as owner:
        owner.execute("TRUNCATE cache.rerank_scores")
    with db.connect("kean-s") as conn:
        conn.execute("INSERT INTO cache.rerank_scores VALUES ('m', 'r', 'q', 't', 1)")
        conn.commit()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(change)
    with db.connect() as owner:
        assert owner.execute("SELECT score FROM cache.rerank_scores").fetchall() == [(1.0,)]


@pytest.mark.usefixtures("store")
def test_readers_never_see_the_embedding_cache() -> None:
    with db.connect("investigator") as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("SELECT count(*) FROM cache.embeddings")
