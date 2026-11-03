"""Documents per topic per day, recounted from `assignments`, never incremented (rule 5)."""

from collections.abc import Iterable
from datetime import date

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.trends.topics import Pool


def recount(conn: psycopg.Connection[TupleRow], days: Iterable[date]) -> None:
    """Rewrite `topic_days` for these days from what's assigned so far."""
    touched = sorted(set(days))
    conn.execute("DELETE FROM topic_days WHERE day = ANY(%s)", (touched,))
    conn.execute(
        "INSERT INTO topic_days (topic_id, day, documents, press)"
        " SELECT topic_id, sent_on, count(DISTINCT document_id),"
        " count(DISTINCT document_id) FILTER (WHERE press) FROM assignments"
        " WHERE sent_on = ANY(%s) GROUP BY 1, 2",
        (touched,),
    )


def record_totals(conn: psycopg.Connection[TupleRow], pool: Pool) -> None:
    """Write what each day's shares divide by: its documents that count, with a topic or not."""
    documents: dict[date, set[int]] = {}
    for day, document in zip(pool.days, pool.document_ids.tolist(), strict=True):
        documents.setdefault(day, set()).add(document)
    conn.cursor().executemany(
        "INSERT INTO day_totals (day, documents) VALUES (%s, %s)"
        " ON CONFLICT (day) DO UPDATE SET documents = EXCLUDED.documents",
        [(day, len(held)) for day, held in documents.items()],
    )
