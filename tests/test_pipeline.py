from datetime import date
from typing import Any, cast

import numpy as np
import psycopg
import pytest

from rag_trend_radar.ingest.kinds import Source
from rag_trend_radar.schema import SearchHit, SearchStage
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


def _reranked(monkeypatch: pytest.MonkeyPatch, stages: list[SearchStage] | None) -> list[SearchHit]:
    found = [_hit(1, _mail(1), 0.9), _hit(2, "Subject: x\nFrom: a@enron.com\n\n(AP) - wire", 0.8)]
    monkeypatch.setattr(hybrid, "search", lambda *_args: found)
    monkeypatch.setattr(pipeline, "rerank", lambda _q, hits, _m, top: list(hits)[:top])
    return pipeline.reranked(
        cast(psycopg.Connection[Any], None), "q", np.zeros(4, dtype=np.float32), "m",
        date(2001, 9, 1), "r", 50, keep=Source.MAIL, stages=stages,
    )  # fmt: skip


def test_stages_count_candidates_before_the_source_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[SearchStage] = []
    ranked = _reranked(monkeypatch, stages)
    assert [s.name for s in stages] == ["search", "rerank"]
    assert [s.items for s in stages] == [2, len(ranked)]


def test_stages_are_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    assert [h.chunk_id for h in _reranked(monkeypatch, None)] == [1]
