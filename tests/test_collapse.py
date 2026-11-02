from hypothesis import given, settings
from hypothesis import strategies as st

from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search.collapse import collapse

ARTICLE = (
    "Dynegy abandoned its plan to rescue Enron after the credit ratings were cut to junk status "
    "on Wednesday raising the prospect of a bankruptcy filing for the energy trader in Houston"
)


def _hit(chunk_id: int, body: str, lead: str = "Subject: News") -> SearchHit:
    return SearchHit(chunk_id=chunk_id, document_id=chunk_id, text=f"{lead}\n\n{body}", score=1.0)


def _ids(hits: list[SearchHit]) -> list[int]:
    return [hit.chunk_id for hit in hits]


def test_an_article_pasted_into_another_digest_counts_once_under_any_lead() -> None:
    hits = [_hit(1, ARTICLE), _hit(2, ARTICLE, "Subject: Enron Mentions\nFrom: a@enron.com")]
    assert _ids(collapse(hits, 10)) == [1]


def test_a_chunk_holding_the_whole_of_a_shorter_one_is_its_copy() -> None:
    longer = f"{ARTICLE} Analysts said the deal had been in doubt for several days already"
    assert _ids(collapse([_hit(1, longer), _hit(2, ARTICLE)], 10)) == [1]


def test_a_report_with_a_different_figure_stays() -> None:
    report = f"{ARTICLE} The wire was for {{}} dollars sent to Entergy Koch"
    hits = [_hit(1, report.format("209139")), _hit(2, report.format("355000"))]
    assert _ids(collapse(hits, 10)) == [1, 2]


def test_different_mail_stays() -> None:
    other = (
        "Please send the signed confidentiality agreement to the legal group before the Friday "
        "meeting so that we can review the remaining open points with outside counsel next week"
    )
    assert _ids(collapse([_hit(1, ARTICLE), _hit(2, other)], 10)) == [1, 2]


def test_short_mail_needs_equal_text() -> None:
    hits = [_hit(1, "Yes, approved."), _hit(2, "Yes, approved."), _hit(3, "Not approved.")]
    assert _ids(collapse(hits, 10)) == [1, 3]


def test_it_stops_at_top_counting_only_kept_hits() -> None:
    hits = [_hit(1, ARTICLE), _hit(2, ARTICLE), _hit(3, "ok"), _hit(4, "fine")]
    assert _ids(collapse(hits, 2)) == [1, 3]


@settings(max_examples=50)
@given(st.lists(st.sampled_from(["ok", "fine", ARTICLE, f"{ARTICLE} again"]), max_size=8))
def test_collapse_keeps_order_and_settles_after_one_pass(bodies: list[str]) -> None:
    hits = [_hit(n, body) for n, body in enumerate(bodies)]
    once = collapse(hits, 10)
    assert _ids(once) == sorted(_ids(once))
    assert collapse(once, 10) == once
