from datetime import date
from typing import Any, cast

import numpy as np
import psycopg
import pytest

from rag_trend_radar.config import get_settings
from rag_trend_radar.ingest.origin import Origin
from rag_trend_radar.records import Ranked, SearchHit
from rag_trend_radar.search import hybrid, pipeline
from rag_trend_radar.search.mix import RESULTS
from rag_trend_radar.search.pipeline import final


def _hit(chunk_id: int, text: str, score: float) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, document_id=chunk_id, text=text, score=score)


def _mail(n: int) -> str:
    return (
        f"Subject: topic {n}\nFrom: a{n}@enron.com\n\nword{n} other{n} thing{n} about{n} deal{n}."
    )


def test_a_repeat_of_a_better_result_drops() -> None:
    hits = [_hit(1, _mail(1), 0.9), _hit(2, _mail(1), 0.8), _hit(3, _mail(3), 0.7)]
    assert [h.chunk_id for h in final(hits, 50)] == [1, 3]


def test_no_more_results_than_reach_claude() -> None:
    hits = [_hit(n, _mail(n), 1 - n / 100) for n in range(1, RESULTS + 6)]
    assert len(final(hits, 50)) == RESULTS


def _reranked(monkeypatch: pytest.MonkeyPatch, **narrow: Any) -> tuple[Ranked, dict[str, str]]:
    """Run reranked on two found chunks, one press; return it and the models it asked for."""
    found = [_hit(1, _mail(1), 0.9), _hit(2, "Subject: x\nFrom: a@enron.com\n\n(AP) - wire", 0.8)]
    asked: dict[str, str] = {}

    def search(_conn: object, _q: str, _v: object, model: str, *_rest: object) -> list[SearchHit]:
        asked["model"] = model
        return found

    def rerank(
        _conn: object, _q: str, hits: list[SearchHit], model: str, top: int
    ) -> list[SearchHit]:
        asked["reranker"] = model
        return hits[:top]

    monkeypatch.setattr(hybrid, "search", search)
    monkeypatch.setattr(pipeline, "rerank", rerank)
    ranked = pipeline.reranked(
        cast(psycopg.Connection[Any], None), "q", np.zeros(4, dtype=np.float32), date(2001, 9, 1),
        **narrow,
    )  # fmt: skip
    return ranked, asked


def test_stages_count_candidates_before_the_origin_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    ranked, _ = _reranked(monkeypatch, origin=Origin.MAIL)
    assert [h.chunk_id for h in ranked.hits] == [1]
    assert [s.name for s in ranked.stages] == ["search", "rerank"]
    assert [s.items for s in ranked.stages] == [2, 1]


def test_the_configured_models_unless_others_are_given(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    _, asked = _reranked(monkeypatch)
    assert asked == {"model": settings.embedding_model, "reranker": settings.rerank_model}
    _, asked = _reranked(monkeypatch, model="other", reranker="other-reranker")
    assert asked == {"model": "other", "reranker": "other-reranker"}
