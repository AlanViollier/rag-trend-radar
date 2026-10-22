"""BM25 search over chunk text (pg_search)."""

from datetime import date

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit


def search(conn: psycopg.Connection[TupleRow], query: str, as_of: date, k: int) -> list[SearchHit]:
    """Top `k` chunks by BM25 among documents sent on or before `as_of`."""
    raise NotImplementedError
