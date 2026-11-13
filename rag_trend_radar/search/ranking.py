"""What every search shares: ranked chunk ids turned into hits."""

from collections.abc import Mapping

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.records import SearchHit

# A document `d` the reader could have read by `cutoff`, on top of row-level security: it says who
# may read a quoted message, not since when, and that needs the day asked.
SEEN = sql.SQL(
    "d.sent_at < %(cutoff)s"
    " AND (d.kind <> 'quoted' OR current_setting('app.person', true) = 'investigator'"
    " OR EXISTS (SELECT 1 FROM access a WHERE a.document_id = d.id"
    " AND a.mailbox = current_setting('app.person', true) AND a.from_at < %(cutoff)s))"
)


def hits(
    conn: psycopg.Connection[TupleRow], ranked: sql.Composable, params: Mapping[str, object]
) -> list[SearchHit]:
    """Run a query yielding (id, score) and attach each chunk's document and text, best first."""
    rows = conn.execute(
        sql.SQL(
            "SELECT r.id, c.document_id, left(c.text, length(c.text) - c.quote_chars),"
            " right(c.text, greatest(c.quote_chars - 2, 0)), r.score FROM ({}) r"
            " JOIN chunks c ON c.id = r.id ORDER BY r.score DESC, r.id"
        ).format(ranked),
        params,
    ).fetchall()
    return [
        SearchHit(chunk_id=i, document_id=d, text=t, context=q, score=float(s))
        for i, d, t, q, s in rows
    ]
