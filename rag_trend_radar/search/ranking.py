"""What every search shares: the `as_of` cutoff, and ranked chunk ids turned into hits."""

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.schema import SearchHit


def cutoff(as_of: date) -> datetime:
    """The instant `as_of` ends in the store's local days: a search sees only mail sent before."""
    return datetime.combine(as_of + timedelta(days=1), time(), ZoneInfo(get_settings().timezone))


def hits(
    conn: psycopg.Connection[TupleRow], ranked: sql.Composable, params: Mapping[str, object]
) -> list[SearchHit]:
    """Run a query yielding (id, score) and attach each chunk's document and text, best first."""
    rows = conn.execute(
        sql.SQL(
            "SELECT r.id, c.document_id, c.text, r.score FROM ({}) r"
            " JOIN chunks c ON c.id = r.id ORDER BY r.score DESC, r.id"
        ).format(ranked),
        params,
    ).fetchall()
    return [SearchHit(chunk_id=i, document_id=d, text=t, score=float(s)) for i, d, t, s in rows]
