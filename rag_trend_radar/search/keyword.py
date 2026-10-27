"""BM25 search over chunk text (pg_search)."""

from datetime import date

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search.ranking import cutoff, hits

## `|||` matches any of the words, unparsed: a question's `?` or `:` isn't query syntax.
## Row-level security filters inside the BM25 scan, before the LIMIT (checked by EXPLAIN).
RANKED = sql.SQL(
    "SELECT c.id, paradedb.score(c.id) AS score FROM chunks c"
    " JOIN documents d ON d.id = c.document_id"
    " WHERE c.text ||| %(query)s AND d.sent_at < %(cutoff)s"
    " ORDER BY score DESC LIMIT %(depth)s"
)


def search(conn: psycopg.Connection[TupleRow], query: str, as_of: date, k: int) -> list[SearchHit]:
    """Top `k` chunks by BM25 among documents sent on or before `as_of`."""
    return hits(conn, RANKED, {"query": query, "cutoff": cutoff(as_of), "depth": k})
