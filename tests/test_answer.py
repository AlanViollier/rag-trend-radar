from rag_trend_radar.answer import CONTEXT_MARK, _search_result
from rag_trend_radar.schema import SearchHit


def test_a_chunk_becomes_one_citable_block_per_paragraph() -> None:
    hit = SearchHit(
        chunk_id=7,
        document_id=3,
        text="Subject: Gas\nFrom: pat.rowe@enron.com\nDate: 2001-09-03\n\nPrices rose.\n\n\nCall me.",
        score=0.5,
    )
    block = _search_result(hit)
    assert (block["title"], block["source"]) == ("Gas", "document 3")
    assert [c["text"] for c in block["content"]] == [
        "Subject: Gas\nFrom: pat.rowe@enron.com\nDate: 2001-09-03",
        "Prices rose.",
        "Call me.",
    ]


def test_a_short_replys_quote_is_its_last_citable_block() -> None:
    hit = SearchHit(
        chunk_id=7,
        document_id=3,
        text="Subject: Budget\n\nYes, approved.",
        score=0.5,
        context="Do we approve the budget?",
    )
    assert [c["text"] for c in _search_result(hit)["content"]][-1] == (
        f"{CONTEXT_MARK} Do we approve the budget?"
    )


def test_a_hit_with_no_quote_gets_no_extra_block() -> None:
    hit = SearchHit(chunk_id=7, document_id=3, text="Subject: Budget\n\nNo.", score=0.5)
    assert len(list(_search_result(hit)["content"])) == 2
