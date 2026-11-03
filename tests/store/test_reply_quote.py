from datetime import UTC, datetime

import pytest

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.ingest.chunk import pieces
from rag_trend_radar.schema import Document
from rag_trend_radar.search import keyword

ABOVE = "-----Original Message-----\nFrom: Rowe, Pat\nSubject: Tickets\n\nPlease send the tickets to Judge Wombat."
DAY = datetime(2001, 9, 3, tzinfo=UTC)


def _document(external_id: str, body: str, quoted: str) -> Document:
    return Document(
        source="test-reply",
        external_id=external_id,
        kind="email",
        sent_at=DAY,
        title=external_id,
        author="",
        body=body,
        quoted=quoted,
        public=False,
        readers=("mann-k",),
    )


def _pieces(doc: Document) -> list[str]:
    return pieces(doc, len, frozenset())


@pytest.fixture
def documents(store: None) -> list[Document]:
    docs = [_document("reply", "i sent it today", ABOVE), _document("alone", "Lunch wombat?", "")]
    with db.connect() as conn:
        conn.execute("INSERT INTO employees VALUES ('mann-k') ON CONFLICT DO NOTHING")
        index.upsert_documents(conn, docs, _pieces)
    return docs


def test_a_short_reply_is_found_by_its_quote_and_keeps_its_own_words_apart(
    documents: list[Document],
) -> None:
    assert documents
    with db.connect("mann-k") as conn:
        hits = keyword.search(conn, "judge", DAY.date(), 5)
    assert [hit.context for hit in hits] == ["Please send the tickets to Judge Wombat."]
    assert hits[0].text.endswith("\n\ni sent it today")
    assert hits[0].full_text.endswith("i sent it today\n\nPlease send the tickets to Judge Wombat.")


def test_a_short_reply_is_tagged_and_other_mail_isnt(documents: list[Document]) -> None:
    assert documents
    with db.connect() as conn:
        tags: dict[str, bool] = dict(
            conn.execute(
                "SELECT d.external_id, c.quote_chars > 0 FROM chunks c"
                " JOIN documents d ON d.id = c.document_id WHERE d.source = 'test-reply'"
            ).fetchall()
        )
    assert tags == {"reply": True, "alone": False}


def test_running_a_reply_again_writes_nothing(documents: list[Document]) -> None:
    with db.connect() as conn:
        assert index.upsert_documents(conn, documents, _pieces) == 0
