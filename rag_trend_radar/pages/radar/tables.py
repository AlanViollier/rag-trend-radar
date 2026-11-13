"""What page 1 reads from the store: every cluster week, the alerts, the days, the glow's sample."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.days import cutoff, opening
from rag_trend_radar.index import vectors_identifier
from rag_trend_radar.ingest.origin import Origin, origin

# A sample of the chunks no cluster ever took, enough for a glow. Bulk mail is dropped after
# sampling, as the radar drops it, so more are read than kept.
_DUST, _DUST_READ = 4000, 6000
_DUST_WEEKS = 26


@dataclass(frozen=True)
class Seen:
    """A cluster in one week, as its table row holds it."""

    cluster: int
    shown: bool
    week: date
    centre: NDArray[np.float32]
    words: tuple[str, ...]
    mail: int
    recent: int
    surge: float


@dataclass(frozen=True)
class Fired:
    """An alert's row, with when its cluster was born."""

    day: date
    cluster: int
    observed: float
    expected: float
    born: date


@dataclass(frozen=True)
class Day:
    """One day's row of `day_totals`."""

    day: date
    documents: int
    chunks: int
    senders: int


@dataclass(frozen=True)
class Speck:
    """A chunk no cluster took, and the day it was sent."""

    vector: NDArray[np.float32]
    day: date


@dataclass(frozen=True)
class Tables:
    """Everything the page reads from the store."""

    seen: list[Seen]
    fired: list[Fired]
    days: list[Day]
    documents_before: int
    chunks_before: int
    dust: list[Speck]


def read(conn: psycopg.Connection[TupleRow], model: str, dims: int, ends: Sequence[date]) -> Tables:
    """Every cluster, the alerts, and the store day by day."""
    seen = [
        Seen(b, s, w, c.to_numpy(), tuple(k), m, r, u)
        for b, s, w, c, k, m, r, u in conn.execute(
            "SELECT b.id, b.shown, w.week, w.centre, w.keywords, w.mail, w.recent, w.surge"
            " FROM clusters b JOIN cluster_weeks w ON w.cluster_id = b.id ORDER BY b.id, w.week"
        ).fetchall()
    ]
    fired = [
        Fired(*row)
        for row in conn.execute(
            "SELECT a.day, a.cluster_id, a.observed, a.expected, b.first_week FROM alerts a"
            " JOIN clusters b ON b.id = a.cluster_id ORDER BY a.day, a.cluster_id"
        ).fetchall()
    ]
    days = [
        Day(*row)
        for row in conn.execute(
            "SELECT day, documents, chunks, senders FROM day_totals ORDER BY day"
        ).fetchall()
    ]
    start = opening(days[0].day)
    documents = conn.execute(
        "SELECT count(*) FROM documents WHERE sent_at < %s", (start,)
    ).fetchall()[0][0]
    chunks = conn.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.sent_at < %s",
        (start,),
    ).fetchall()[0][0]
    return Tables(seen, fired, days, documents, chunks, _dust(conn, model, dims, ends))


def _dust(
    conn: psycopg.Connection[TupleRow], model: str, dims: int, ends: Sequence[date]
) -> list[Speck]:
    rows = conn.execute(
        sql.SQL(
            "SELECT v.embedding, (d.sent_at AT TIME ZONE %s)::date, c.text FROM chunks c"
            " JOIN documents d ON d.id = c.document_id JOIN {} v ON v.chunk_id = c.id"
            " WHERE d.sent_at >= %s AND d.sent_at < %s AND c.quote_chars = 0"
            " AND NOT EXISTS (SELECT 1 FROM cluster_chunks b WHERE b.chunk_id = c.id)"
            " ORDER BY c.text_hash LIMIT %s"
        ).format(vectors_identifier(model, dims)),
        (
            get_settings().timezone,
            cutoff(ends[0] - timedelta(weeks=_DUST_WEEKS)),
            cutoff(ends[-1]),
            _DUST_READ,
        ),
    ).fetchall()
    return [Speck(v.to_numpy(), d) for v, d, text in rows if origin(text) is not Origin.BULK][
        :_DUST
    ]
