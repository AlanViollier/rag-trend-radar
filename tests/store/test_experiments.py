from collections.abc import Iterator

import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar import index
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.ingest.dedup import text_hash


@pytest.fixture
def experiment(store: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with db.connect() as conn:
        conn.execute("INSERT INTO employees VALUES ('in-public') ON CONFLICT DO NOTHING")
    monkeypatch.setenv("DB_SCHEMA", "exp_isolation")
    get_settings.cache_clear()
    db.reset()
    yield
    monkeypatch.setenv("DB_SCHEMA", "public")
    get_settings.cache_clear()


@pytest.mark.usefixtures("experiment")
def test_an_experiment_gets_fresh_tables_and_leaves_the_store_alone() -> None:
    with db.connect() as conn:
        assert conn.execute("SELECT count(*) FROM employees").fetchall() == [(0,)]
        conn.execute("INSERT INTO employees VALUES ('in-experiment')")
        assert conn.execute(
            "SELECT mailbox FROM public.employees WHERE mailbox = 'in-public'"
        ).fetchall() == [("in-public",)]


@pytest.mark.usefixtures("experiment")
def test_readers_can_search_inside_an_experiment() -> None:
    with db.connect("investigator") as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchall() == [(0,)]


@pytest.mark.usefixtures("experiment")
def test_vectors_go_to_the_experiment_even_when_the_store_has_the_table() -> None:
    with db.connect() as conn:
        # The store has one already: an unqualified name would resolve to it and skip creation.
        conn.execute("CREATE TABLE IF NOT EXISTS public.vectors_fake_8 (chunk_id bigint)")
        index.write_vectors(conn, "test/fake", [], np.zeros((0, 8), dtype=np.float32))
        tables = conn.execute(
            "SELECT schemaname FROM pg_tables WHERE tablename = 'vectors_fake_8' ORDER BY 1"
        ).fetchall()
    assert ("exp_isolation",) in tables


@pytest.mark.usefixtures("experiment")
def test_a_vector_outlives_a_reset_and_comes_back_for_the_same_text() -> None:
    with db.connect() as conn:
        conn.execute("INSERT INTO employees VALUES ('lee-p') ON CONFLICT DO NOTHING")
        before = _chunk(conn, "exp_isolation", "same text")
        index.write_vectors(conn, "test/reuse", [before], np.eye(4, dtype=np.float32)[:1])
        conn.commit()
    db.reset()
    with db.connect() as conn:
        _chunk(conn, "exp_isolation", "same text")
        new = _chunk(conn, "exp_isolation", "new text")
        assert index.from_cache(conn, "test/reuse", 4) == 1
        assert index.unembedded(conn, "test/reuse", 4) == {"new text": [new]}
        conn.rollback()
        conn.execute("DELETE FROM cache.embeddings WHERE model = 'test/reuse'")
        conn.commit()


def _chunk(conn: psycopg.Connection[TupleRow], schema: str, text: str) -> int:
    document = conn.execute(
        sql.SQL(
            "INSERT INTO {} (source, external_id, kind, sent_at, title, body)"
            " VALUES ('test-reuse', %s, 'email', now(), '', %s) RETURNING id"
        ).format(sql.Identifier(schema, "documents")),
        (f"{schema}-{text}", text),
    ).fetchall()[0][0]
    return int(
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (document_id, position, text, text_hash) VALUES (%s, 0, %s, %s)"
                " RETURNING id"
            ).format(sql.Identifier(schema, "chunks")),
            (document, text, text_hash(text)),
        ).fetchall()[0][0]
    )


@pytest.mark.usefixtures("store")
def test_a_reset_leaves_tables_an_extension_owns() -> None:
    # A fresh image installs PostGIS in every new database: `db-reset` crashed on its table.
    with db.connect() as conn:
        had_it = conn.execute("SELECT to_regclass('public.spatial_ref_sys')").fetchall()[0][0]
        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    try:
        db.reset()
        with db.connect() as conn:
            assert conn.execute("SELECT to_regclass('public.spatial_ref_sys')").fetchall()[0][0]
            assert conn.execute("SELECT count(*) FROM employees").fetchall() == [(0,)]
    finally:
        if had_it is None:
            with db.connect() as conn:
                conn.execute("DROP EXTENSION postgis")


@pytest.mark.usefixtures("experiment")
def test_a_reset_makes_a_new_store_so_saved_chunk_ids_cant_carry_over() -> None:
    with db.connect() as conn:
        before = conn.execute("SELECT id FROM store").fetchall()
    db.reset()
    with db.connect() as conn:
        after = conn.execute("SELECT id FROM store").fetchall()
    assert len(before) == len(after) == 1
    assert before != after
