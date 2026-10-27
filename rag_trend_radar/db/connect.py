"""Connections to the store, as the pipeline or as a person who is searching."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
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
        conn.commit()
    return conn


def reset() -> None:
    """Drop everything and recreate the tables from schema.sql."""
    with _open() as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        conn.execute(SCHEMA_SQL.read_text())


def _open() -> psycopg.Connection[TupleRow]:
    settings = get_settings()
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.postgres_password.get_secret_value(),
    )
