from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
import pytest
from psycopg.rows import TupleRow

from rag_trend_radar import batch, index
from rag_trend_radar.db import connect as db
from rag_trend_radar.ingest.chunk import pieces
from rag_trend_radar.schema import Document

CHICAGO = ZoneInfo("America/Chicago")
MODEL, DIMS = "test/fake", 8
QUOTE = "-----Original Message-----\nFrom: Kit Moss\nSent: Friday\nTo: Lee Park\nSubject: Gas\n\n"
Connection = psycopg.Connection[TupleRow]


@pytest.fixture
def conn(store: None) -> Iterator[Connection]:
    with db.connect() as conn:
        yield conn
        conn.rollback()
        for table in ("quotes", "copies", "boilerplate", "quarantine", "raw_files", "batches"):
            conn.execute(f"DELETE FROM {table}")  # noqa: S608 (fixed names)
        conn.execute("DELETE FROM documents WHERE source = 'enron'")
        conn.commit()


def _write(root: Path, path: str, sent: str, subject: str, body: str) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        f"Message-ID: <{path}>\nDate: {sent}\nFrom: someone@enron.com\nTo: lee.park@enron.com\n"
        f"Subject: {subject}\n\n{body}"
    )


def _pieces(doc: Document, frequent: frozenset[str]) -> list[str]:
    return pieces(doc, lambda text: len(text.split()), frequent)


def _run(conn: Connection, root: Path, *days: date) -> None:
    batch.stage(conn, root, CHICAGO)
    for day in days:
        batch.run_day(conn, day, CHICAGO, _pieces)
        batch.finish_day(conn, day, MODEL, DIMS)


def _documents(conn: Connection) -> dict[str, tuple[str, datetime, list[str]]]:
    rows = conn.execute(
        "SELECT d.title, d.kind, d.sent_at, array_agg(a.mailbox ORDER BY a.mailbox)"
        " FROM documents d LEFT JOIN access a ON a.document_id = d.id"
        " WHERE d.source = 'enron' GROUP BY d.id"
    ).fetchall()
    return {title: (kind, sent_at, readers) for title, kind, sent_at, readers in rows}


def test_copies_merge_into_one_email_readable_from_every_mailbox(
    conn: Connection, tmp_path: Path
) -> None:
    _write(tmp_path, "kim-d/inbox/1.", "Mon, 10 Sep 2001 09:00:00 -0500", "plan", "at 3")
    _write(tmp_path, "rowe-p/sent/7.", "Mon, 10 Sep 2001 09:00:00 -0500", "plan", "at  3\n")
    _run(conn, tmp_path, date(2001, 9, 10))
    assert _documents(conn)["plan"][2] == ["kim-d", "rowe-p"]


def test_a_day_holds_the_emails_whose_resolved_date_falls_on_it(
    conn: Connection, tmp_path: Path
) -> None:
    ## 23:30 on Aug 31 in Chicago by one export, 06:30:12 Sep 1 by the server copy, which wins.
    _write(tmp_path, "kim-d/inbox/2.", "Fri, 31 Aug 2001 23:30 -0500", "edge", "x")
    _write(tmp_path, "rowe-p/inbox/2.", "Sat, 1 Sep 2001 06:30:12 -0500", "edge", "x")
    _write(tmp_path, "kim-d/inbox/3.", "Fri, 31 Aug 2001 22:00:00 -0500", "august", "y")
    _run(conn, tmp_path, date(2001, 8, 31))
    assert "edge" not in _documents(conn)
    _run(conn, tmp_path, date(2001, 9, 1))
    documents = _documents(conn)
    assert documents["edge"][1] == datetime(2001, 9, 1, 11, 30, 12, tzinfo=UTC)
    assert documents["august"][1] == datetime(2001, 9, 1, 3, 0, tzinfo=UTC)


def test_a_quoted_message_quoted_on_two_days_is_one_document_for_both_readers(
    conn: Connection, tmp_path: Path
) -> None:
    _write(
        tmp_path,
        "kim-d/inbox/4.",
        "Mon, 3 Sep 2001 10:00 -0500",
        "RE: Gas",
        "ok\n\n" + QUOTE + "Up.",
    )
    _write(
        tmp_path,
        "park-l/inbox/5.",
        "Tue, 4 Sep 2001 10:00 -0500",
        "FW: Gas",
        "fyi\n\n" + QUOTE + "Up.",
    )
    _run(conn, tmp_path, date(2001, 9, 3))
    assert _documents(conn)["Gas"] == (
        "quoted",
        datetime(2001, 9, 3, 15, 0, tzinfo=UTC),
        ["kim-d"],
    )
    batch.run_day(conn, date(2001, 9, 4), CHICAGO, _pieces)
    assert _documents(conn)["Gas"][1:] == (
        datetime(2001, 9, 3, 15, 0, tzinfo=UTC),
        ["kim-d", "park-l"],
    )


def test_a_quote_of_an_email_held_on_another_day_adds_no_document(
    conn: Connection, tmp_path: Path
) -> None:
    _write(tmp_path, "kim-d/sent/6.", "Mon, 3 Sep 2001 10:00 -0500", "Gas", "Prices rose.")
    _write(
        tmp_path,
        "park-l/inbox/6.",
        "Tue, 4 Sep 2001 11:00 -0500",
        "RE: Gas",
        "ok\n\n" + QUOTE + "Prices rose.",
    )
    _run(conn, tmp_path, date(2001, 9, 4))
    assert [kind for kind, _, _ in _documents(conn).values()] == ["email"]


def test_junk_dates_are_quarantined_and_kept(conn: Connection, tmp_path: Path) -> None:
    _write(tmp_path, "kim-d/inbox/8.", "Mon, 31 Dec 1979 16:00:00 -0800", "old", "z")
    _run(conn, tmp_path)
    assert conn.execute("SELECT path FROM raw_files").fetchall() == [("kim-d/inbox/8.",)]
    assert conn.execute("SELECT reason, detail FROM quarantine").fetchall() == [
        ("date outside 1997–2002", "Mon, 31 Dec 1979 16:00:00 -0800")
    ]


def test_running_a_day_again_changes_nothing(conn: Connection, tmp_path: Path) -> None:
    _write(
        tmp_path,
        "kim-d/inbox/4.",
        "Mon, 3 Sep 2001 10:00 -0500",
        "RE: Gas",
        "ok\n\n" + QUOTE + "Up.",
    )
    _write(
        tmp_path,
        "park-l/inbox/5.",
        "Tue, 4 Sep 2001 10:00 -0500",
        "FW: Gas",
        "fyi\n\n" + QUOTE + "Up.",
    )
    _write(tmp_path, "kim-d/sent/9.", "Tue, 4 Sep 2001 23:30 -0500", "late", "after hours")
    _run(conn, tmp_path, date(2001, 9, 3), date(2001, 9, 4))
    before = _snapshot(conn)
    _run(conn, tmp_path, date(2001, 9, 3), date(2001, 9, 4), date(2001, 9, 3))
    assert _snapshot(conn) == before
    assert before["batches"] == [(date(2001, 9, 3), 1, 2, 0), (date(2001, 9, 4), 2, 2, 0)]


def _snapshot(conn: Connection) -> dict[str, list[tuple[object, ...]]]:
    queries = {
        "documents": "SELECT id, external_id, kind, sent_at, title, body FROM documents"
        " WHERE source = 'enron' ORDER BY id",
        "access": "SELECT document_id, mailbox FROM access ORDER BY 1, 2",
        "chunks": "SELECT id, document_id, position, text FROM chunks ORDER BY id",
        "quotes": "SELECT email_id, quoted_id FROM quotes ORDER BY 1, 2",
        "batches": "SELECT day, emails_in, documents_new, chunks_embedded FROM batches"
        " WHERE finished_at IS NOT NULL ORDER BY day",
    }
    return {name: conn.execute(query).fetchall() for name, query in queries.items()}


def test_a_deleted_email_stays_gone_and_takes_only_the_quotes_it_alone_held(
    conn: Connection, tmp_path: Path
) -> None:
    oil = QUOTE.replace("Subject: Gas", "Subject: Oil")
    _write(
        tmp_path,
        "kim-d/inbox/4.",
        "Mon, 3 Sep 2001 10:00 -0500",
        "RE: Gas",
        "ok\n\n" + QUOTE + "Up.",
    )
    _write(
        tmp_path,
        "kim-d/inbox/5.",
        "Mon, 3 Sep 2001 11:00 -0500",
        "RE: Oil",
        "no\n\n" + oil + "Down.",
    )
    _write(
        tmp_path,
        "park-l/inbox/6.",
        "Tue, 4 Sep 2001 10:00 -0500",
        "FW: Gas",
        "fyi\n\n" + QUOTE + "Up.",
    )
    _run(conn, tmp_path, date(2001, 9, 3), date(2001, 9, 4))
    ids: dict[str, int] = dict(
        conn.execute("SELECT title, id FROM documents WHERE source = 'enron'").fetchall()
    )

    for title in ("RE: Gas", "RE: Oil"):
        for sent_at in index.delete_document(conn, ids[title]):
            batch.recount_day(conn, sent_at.astimezone(CHICAGO).date(), MODEL, DIMS)
    batch.run_day(conn, date(2001, 9, 3), CHICAGO, _pieces)

    deleted = conn.execute(
        "SELECT id FROM documents WHERE deleted_at IS NOT NULL AND body = '' ORDER BY id"
    ).fetchall()
    assert deleted == sorted([(ids["RE: Gas"],), (ids["RE: Oil"],), (ids["Oil"],)])
    assert conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id = ANY(%s)", ([i for (i,) in deleted],)
    ).fetchall() == [(0,)]
    assert _documents(conn)["Gas"][2] == ["park-l"]
    assert conn.execute("SELECT emails_in FROM batches WHERE day = '2001-09-03'").fetchall() == [
        (0,)
    ]
    conn.commit()  # the reader is another connection
    with db.connect("investigator") as reader:
        visible = reader.execute("SELECT count(*) FROM documents WHERE source = 'enron'").fetchall()
    assert visible == [(2,)]  # FW: Gas and the Gas it quotes
