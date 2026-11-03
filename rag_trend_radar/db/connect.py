"""Connections to the store, as the pipeline or as a person who is searching."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings

SCHEMA_SQL = Path(__file__).with_name("schema.sql")
INVESTIGATOR = "investigator"
## The graph walk keeps this many candidates (default 40). A filter drops most of them, so at 40
## the right email was missing from the top 50 for 13% of questions; at 1,000 (the maximum)
## recall matched exact search, at the same ~40 ms (2026-10-27, Sep 2001).
EF_SEARCH = 1000
## A filtered walk stops here even with too few rows (default 20,000): the investigator asking
## as of mid-1999, when ~1% of the mail exists, got 33 of 50 (2026-11-02).
MAX_SCAN_TUPLES = 100_000


def connect(person: str | None = None, ef_search: int = EF_SEARCH) -> psycopg.Connection[TupleRow]:
    """Open a connection; with `person`, every query sees only what that person may see.

    Parameters
    ----------
    person:
        None for the pipeline (owner, no row filtering). 'investigator' or an employee's
        mailbox (kean-s) for searches: the connection switches to radar_reader and sets
        app.person.
    ef_search:
        How many candidates a reader's HNSW scan keeps; recall varies it to re-measure.
    """
    conn = _open()
    register_vector(conn)
    if person is not None:
        conn.execute("SET ROLE radar_reader")
        conn.execute("SELECT set_config('app.person', %s, false)", (person,))
        ## An HNSW scan returns its ~40 nearest vectors, then the filters drop most of them.
        ## Iterative scans keep walking the graph until enough rows pass, in exact distance
        ## order (pgvector 0.8).
        conn.execute("SET hnsw.iterative_scan = strict_order")
        conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(ef_search)))
        conn.execute(sql.SQL("SET hnsw.max_scan_tuples = {}").format(sql.Literal(MAX_SCAN_TUPLES)))
        conn.commit()
    return conn


def reset() -> None:
    """Drop the configured schema's tables (public, or an experiment's) and recreate them.

    Tables, not the schema: dropping public would take pgvector with it, and with pgvector
    every vectors table and the embedding cache, hours of embedding (2026-10-29). Tables an
    extension owns stay: a fresh image installs PostGIS, and its table can't be dropped.
    """
    schema = get_settings().db_schema
    with _open() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        tables = conn.execute(
            "SELECT t.tablename FROM pg_tables t WHERE t.schemaname = %s AND NOT EXISTS ("
            " SELECT 1 FROM pg_depend d WHERE d.objid = format('%%I.%%I', t.schemaname, t.tablename)"
            "::regclass AND d.deptype = 'e')",
            (schema,),
        ).fetchall()
        for (table,) in tables:
            conn.execute(sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(schema, table)))
        conn.execute(SCHEMA_SQL.read_text())


def _open() -> psycopg.Connection[TupleRow]:
    settings = get_settings()
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.postgres_password.get_secret_value(),
        ## Unqualified names resolve in the configured schema first; public keeps the extensions.
        options=f"-c search_path={settings.db_schema},public",
    )
