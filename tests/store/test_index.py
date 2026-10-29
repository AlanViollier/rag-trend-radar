from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
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


def _chunks(conn: Connection) -> list[str]:
    rows = conn.execute(
        "SELECT c.text FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.source = 'test-index' ORDER BY c.position"
    ).fetchall()
    return [text for (text,) in rows]


def test_a_new_model_fills_beside_the_old_and_drops_once_swapped(conn: Connection) -> None:
    index.upsert_documents(conn, [_document("a\n\nb")], _pieces)
    first, *_ = (
        i
        for (i,) in conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.source = 'test-index' ORDER BY c.id"
        ).fetchall()
    )
    index.write_vectors(conn, "test/new", [first], np.ones((1, 4), dtype=np.float32))
    assert index.coverage(conn)["vectors_new_4"] == 1
    index.drop_vectors(conn, "test/new", 4)
    assert "vectors_new_4" not in index.coverage(conn)


def test_a_quoted_message_no_email_quotes_any_more_goes(conn: Connection) -> None:
    def day(quoted_text: str) -> list[Document]:
        quoted = _document(quoted_text).model_copy(
            update={
                "external_id": quoted_text,
                "kind": "quoted",
                "readers": (),
                "quoted_by": ("memo",),
            }
        )
        return [_document(f"ok\n\n{quoted_text}"), quoted]

    index.upsert_documents(conn, day("Prices rose."), _pieces)
    index.upsert_documents(conn, day("Prices rose"), _pieces)  # the cleaner now drops the dot
    assert conn.execute(
        "SELECT external_id FROM documents WHERE source = 'test-index' AND kind = 'quoted'"
    ).fetchall() == [("Prices rose",)]


def test_a_change_outside_the_chunks_keeps_them_and_a_change_inside_rewrites_them(
    conn: Connection,
) -> None:
    index.upsert_documents(conn, [_document("a\n\nb")], _pieces)
    ids = _chunk_ids(conn)
    quoted_only = _document("a\n\nb").model_copy(update={"quoted": "> older"})
    assert index.upsert_documents(conn, [quoted_only], _pieces) == 0
    assert _chunk_ids(conn) == ids
    assert index.upsert_documents(conn, [quoted_only], lambda doc: [doc.body]) == 1


def _chunk_ids(conn: Connection) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.source = 'test-index' ORDER BY c.position"
    ).fetchall()
    return [i for (i,) in rows]
