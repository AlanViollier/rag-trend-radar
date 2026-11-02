from datetime import UTC, datetime

import pytest

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import Document, SearchHit
from rag_trend_radar.search.context import QUOTE_CHARS, SHORT_BODY, with_quotes

ABOVE = "Do we approve the Doorstep budget?"


def _document(external_id: str, body: str, quoted: str) -> Document:
    return Document(
        source="test-context",
        external_id=external_id,
        kind="email",
        sent_at=datetime(2001, 9, 3, tzinfo=UTC),
        title=external_id,
        author="",
        body=body,
        quoted=quoted,
        public=False,
        readers=("mann-k",),
    )


@pytest.fixture
def ids(store: None) -> dict[str, int]:
    documents = [
        _document("short", "Yes, approved.", ABOVE),
        _document("long", "x" * SHORT_BODY, ABOVE),
        _document("alone", "Lunch?", ""),
        _document("huge-quote", "Fine.", "y" * (QUOTE_CHARS + 500)),
    ]
    with db.connect() as conn:
        conn.execute("INSERT INTO employees VALUES ('mann-k') ON CONFLICT DO NOTHING")
        index.upsert_documents(conn, documents, lambda doc: [doc.body])
        rows = conn.execute(
            "SELECT external_id, id FROM documents WHERE source = 'test-context'"
        ).fetchall()
    return dict(rows)


def _context(person: str, document_id: int) -> str:
    hit = SearchHit(chunk_id=1, document_id=document_id, text="t", score=0.5)
    with db.connect(person) as conn:
        return with_quotes(conn, [hit])[0].context


def test_a_short_reply_gets_what_it_answered(ids: dict[str, int]) -> None:
    assert _context("mann-k", ids["short"]) == ABOVE


def test_a_long_email_and_one_quoting_nothing_stay_as_they_are(ids: dict[str, int]) -> None:
    assert _context("mann-k", ids["long"]) == ""
    assert _context("mann-k", ids["alone"]) == ""


def test_a_long_quote_is_cut(ids: dict[str, int]) -> None:
    assert len(_context("mann-k", ids["huge-quote"])) == QUOTE_CHARS


def test_a_person_who_cant_read_the_email_gets_no_quote(ids: dict[str, int]) -> None:
    assert _context("rowe-p", ids["short"]) == ""
