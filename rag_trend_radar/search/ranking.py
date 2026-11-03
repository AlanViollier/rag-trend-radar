"""What every search shares: ranked chunk ids turned into hits."""

from collections.abc import Mapping

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit


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
