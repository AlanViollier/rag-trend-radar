"""The radar's tables: investigator-only, and resettable without touching the store."""

import re
from collections.abc import Iterator

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
        cluster = conn.execute(
            "INSERT INTO clusters (id, model, first_week, last_week, kind, shown)"
            " VALUES (1, 'test', '2001-09-09', '2001-09-09', 'topic', true) RETURNING id"
        ).fetchall()[0][0]
        conn.execute(
            "INSERT INTO cluster_weeks VALUES"
            " (%s, '2001-09-09', 0, %s, '{gas}', 1, 1, 1, 0, 0.5, 0.0, true)",
            (cluster, VECTOR),
        )
        conn.execute(
            "INSERT INTO cluster_chunks VALUES (%s, %s, '2001-09-09', '2001-09-09')",
            (cluster, chunk),
        )
        conn.execute("INSERT INTO day_totals VALUES ('2001-09-10', 1, 1, 1, 1, 1, 0)")
        conn.execute(
            "INSERT INTO alerts (day, cluster_id, observed, expected)"
            " VALUES ('2001-09-10', %s, 0.2, 0.1)",
            (cluster,),
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
def test_a_deleted_chunk_leaves_the_cluster() -> None:
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM chunks WHERE document_id IN"
            " (SELECT id FROM documents WHERE source = 'radar-test')"
        )
    assert _count(None, "cluster_chunks") == 0


@pytest.mark.usefixtures("radar")
def test_re_raising_the_same_alert_is_refused() -> None:
    with db.connect() as conn, pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO alerts (day, cluster_id, observed, expected)"
            " SELECT day, cluster_id, observed, expected FROM alerts"
        )
