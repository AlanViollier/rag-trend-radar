"""Connections to the store, as the pipeline or as a person who is searching."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


def connect(person: str | None = None) -> psycopg.Connection[TupleRow]:
    """Open a connection; with `person`, every query sees only what that person may see.

    Parameters
    ----------
    person:
        None for the pipeline (owner, no row filtering). 'investigator' or an employee's
        mailbox (kean-s) for searches: the connection switches to radar_reader and sets
        app.person.
    """
    conn = _open()
    register_vector(conn)
    if person is not None:
        conn.execute("SET ROLE radar_reader")
        conn.execute("SELECT set_config('app.person', %s, false)", (person,))
        ## An HNSW scan returns its ~40 nearest vectors, then row-level security filters them:
        ## an employee who can read 2% of the mail got none back. Iterative scans keep walking
        ## the graph until enough rows pass, in exact distance order (pgvector 0.8).
        conn.execute("SET hnsw.iterative_scan = strict_order")
        ## The graph walk keeps this many candidates (default 40). Most are filtered out by RLS,
        ## so at 40 the right email was missing from the top 50 for 13% of questions; at 1,000
        ## (the maximum) recall matches exact search, at the same ~40 ms (2026-10-27, Sep 2001).
        conn.execute("SET hnsw.ef_search = 1000")
        conn.commit()
    return conn


def reset() -> None:
    """Drop the configured schema's tables (public, or an experiment's) and recreate them.

    Tables, not the schema: dropping public would take pgvector with it, and with pgvector
    every vectors table and the embedding cache, hours of embedding (2026-10-29).
    """
    schema = get_settings().db_schema
    with _open() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        tables = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = %s", (schema,)
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
