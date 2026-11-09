from rag_trend_radar.schema import SearchHit
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
