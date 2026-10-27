from rag_trend_radar.answer import _search_result
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
