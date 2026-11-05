"""The radar's tables: investigator-only, and resettable without touching the store."""

import re
from collections.abc import Iterator
from datetime import date

import psycopg
import pytest
from psycopg import sql

from rag_trend_radar.db import connect as db

VECTOR = "[1,0,0,0]"


@pytest.fixture
def radar(store: None) -> Iterator[None]:
    """One row in every radar table, on one chunk of its own, removed afterwards."""
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO documents (source, external_id, kind, sent_at, title, body)"
            " VALUES ('radar-test', 'one', 'email', '2001-09-10', 'Gas', 'Gas prices.')"
        )
        chunk = conn.execute(
            "INSERT INTO chunks (document_id, position, text, text_hash)"
            " SELECT id, 0, 'Gas prices.', 'radar-test-hash' FROM documents"
            " WHERE source = 'radar-test' RETURNING id"
        ).fetchall()[0][0]
        blob = conn.execute(
            "INSERT INTO blobs (id, model, first_week, last_week, kind, shown)"
            " VALUES (1, 'test', '2001-09-09', '2001-09-09', 'topic', true) RETURNING id"
        ).fetchall()[0][0]
        conn.execute(
            "INSERT INTO blob_weeks VALUES"
            " (%s, '2001-09-09', 0, %s, '{gas}', 1, 1, 1, 0, 0.5, 0.0, true)",
            (blob, VECTOR),
        )
        conn.execute(
            "INSERT INTO blob_chunks VALUES (%s, %s, '2001-09-09', '2001-09-09')", (blob, chunk)
        )
        conn.execute("INSERT INTO day_totals VALUES ('2001-09-10', 1, 1, 1, 1, 1, 0)")
        watchlist = conn.execute(
            "INSERT INTO watchlists (name, query, terms, query_vector, threshold)"
            " VALUES ('worry', 'shredding', '{shred}', %s, 0.5) RETURNING id",
            (VECTOR,),
        ).fetchall()[0][0]
        conn.execute(
            "INSERT INTO alerts (day, kind, blob_id, observed, expected)"
            " VALUES ('2001-09-10', 'spike', %s, 0.2, 0.1)",
            (blob,),
        )
        conn.execute(
            "INSERT INTO alerts (day, kind, watchlist_id, observed)"
            " VALUES ('2001-09-10', 'watchlist', %s, 1)",
            (watchlist,),
        )
    yield
    db.reset_radar()
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = 'radar-test'")


def _count(person: str | None, table: str) -> int:
    with db.connect(person) as conn:
        counted = conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        ).fetchall()[0][0]
    return int(counted)


def test_every_table_in_the_file_is_one_a_radar_reset_drops() -> None:
    assert set(re.findall(r"CREATE TABLE (\w+)", db.RADAR_SQL.read_text())) == set(db.RADAR_TABLES)


@pytest.mark.usefixtures("radar")
@pytest.mark.parametrize("table", db.RADAR_TABLES)
def test_only_the_investigator_reads_the_radar(table: str) -> None:
    assert _count("investigator", table) > 0
    assert _count("kim-d", table) == 0


@pytest.mark.usefixtures("radar")
def test_a_radar_reset_empties_the_radar_and_leaves_the_store() -> None:
    db.reset_radar()
    assert all(_count(None, table) == 0 for table in db.RADAR_TABLES)
    with db.connect() as conn:
        kept = conn.execute(
            "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.source = 'radar-test'"
        ).fetchall()
    assert kept == [(1,)]


@pytest.mark.usefixtures("radar")
def test_a_deleted_chunk_leaves_the_blob() -> None:
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM chunks WHERE document_id IN"
            " (SELECT id FROM documents WHERE source = 'radar-test')"
        )
    assert _count(None, "blob_chunks") == 0


@pytest.mark.usefixtures("radar")
@pytest.mark.parametrize(
    ("kind", "blob", "watchlist"),
    [("spike", False, False), ("spike", True, True), ("watchlist", True, False)],
)
def test_an_alert_points_at_exactly_what_its_kind_says(
    kind: str, blob: bool, watchlist: bool
) -> None:
    with db.connect() as conn:
        blob_id = conn.execute("SELECT id FROM blobs").fetchall()[0][0] if blob else None
        watchlist_id = (
            conn.execute("SELECT id FROM watchlists").fetchall()[0][0] if watchlist else None
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO alerts (day, kind, blob_id, watchlist_id, observed)"
                " VALUES (%s, %s, %s, %s, 1)",
                (date(2001, 9, 11), kind, blob_id, watchlist_id),
            )


@pytest.mark.usefixtures("radar")
def test_re_raising_the_same_alert_is_refused() -> None:
    with db.connect() as conn, pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO alerts (day, kind, blob_id, observed)"
            " SELECT day, kind, blob_id, observed FROM alerts WHERE kind = 'spike'"
        )
