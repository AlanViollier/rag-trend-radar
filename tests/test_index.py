from collections.abc import Iterator
from datetime import date, datetime
from zoneinfo import ZoneInfo

import psycopg
import pytest
from psycopg.rows import TupleRow

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import Document

CHICAGO = ZoneInfo("America/Chicago")
MAILBOXES = ("lee-p", "moss-k")
Connection = psycopg.Connection[TupleRow]


def _document(body: str, readers: tuple[str, ...] = ("lee-p",)) -> Document:
    return Document(
        source="test-index",
        external_id="memo",
        kind="email",
        sent_at=datetime(2001, 9, 5, 9, 0, tzinfo=CHICAGO),
        title="memo",
        author="",
        body=body,
        quoted="",
        body_hash=body,
        public=False,
        readers=readers,
    )


def _pieces(doc: Document) -> list[str]:
    return doc.body.split("\n\n")


@pytest.fixture
def conn(store: None) -> Iterator[Connection]:
    with db.connect() as conn:
        conn.cursor().executemany(
            "INSERT INTO employees (mailbox) VALUES (%s) ON CONFLICT DO NOTHING",
            [(m,) for m in MAILBOXES],
        )
        yield conn
        conn.execute("DELETE FROM documents WHERE source = 'test-index'")


def test_storing_the_same_document_twice_writes_no_chunks_the_second_time(conn: Connection) -> None:
    assert index.upsert_documents(conn, [_document("a\n\nb")], _pieces) == 2
    assert index.upsert_documents(conn, [_document("a\n\nb")], _pieces) == 0
    assert _chunks(conn) == ["a", "b"]


def test_a_changed_body_is_rechunked_in_place(conn: Connection) -> None:
    index.upsert_documents(conn, [_document("a\n\nb")], _pieces)
    assert index.upsert_documents(conn, [_document("c")], _pieces) == 1
    assert _chunks(conn) == ["c"]
    assert conn.execute(
        "SELECT count(*) FROM documents WHERE source = 'test-index'"
    ).fetchall() == [(1,)]


def test_readers_are_rewritten_not_added_to(conn: Connection) -> None:
    index.upsert_documents(conn, [_document("a", readers=("lee-p",))], _pieces)
    index.upsert_documents(conn, [_document("a", readers=("moss-k",))], _pieces)
    assert conn.execute(
        "SELECT mailbox FROM access JOIN documents d ON d.id = access.document_id"
        " WHERE d.source = 'test-index'"
    ).fetchall() == [("moss-k",)]


def test_days_are_recounted_in_local_time(conn: Connection) -> None:
    late = _document("a").model_copy(
        update={"sent_at": datetime(2001, 9, 5, 23, 30, tzinfo=CHICAGO)}
    )
    index.upsert_documents(conn, [late], _pieces)
    index.record_days(conn, date(2001, 9, 5), date(2001, 9, 6), "America/Chicago")
    index.record_days(conn, date(2001, 9, 5), date(2001, 9, 6), "America/Chicago")
    counts = conn.execute(
        "SELECT day, emails_in FROM batches WHERE day IN ('2001-09-05', '2001-09-06') ORDER BY day"
    ).fetchall()
    assert [(d.isoformat(), n) for d, n in counts] == [("2001-09-05", 1), ("2001-09-06", 0)]


def _chunks(conn: Connection) -> list[str]:
    rows = conn.execute(
        "SELECT c.text FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.source = 'test-index' ORDER BY c.position"
    ).fetchall()
    return [text for (text,) in rows]
