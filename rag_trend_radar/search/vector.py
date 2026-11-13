"""Nearest-meaning search over one model's vectors (pgvector HNSW)."""

from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.days import EARLIEST, cutoff
from rag_trend_radar.db import connect as db
from rag_trend_radar.index import vectors_identifier
from rag_trend_radar.records import SearchHit
from rag_trend_radar.search.ranking import hits

# The investigator reads most of the store: a walk of the HNSW index finds enough rows.
INDEXED = sql.SQL(
    "SELECT v.chunk_id AS id, 1 - (v.embedding <=> %(vector)s) AS score FROM {} v"
    " JOIN chunks c ON c.id = v.chunk_id JOIN documents d ON d.id = c.document_id"
    " WHERE d.sent_at >= %(since)s AND d.sent_at < %(cutoff)s ORDER BY v.embedding <=> %(vector)s LIMIT %(depth)s"
)
# An employee reads ~0.5% of it. A walk stops at its scan limits before finding enough of their
# rows (8 of 50 came back for a 595-chunk mailbox, 2026-11-02), while computing the distance to
# each of their chunks misses nothing and takes a median 102 ms against 184. `+ 0` keeps the
# planner off the HNSW index; the reader's rows are listed first, or it scans every chunk.
EXACT = sql.SQL(
    "WITH readable AS MATERIALIZED ("
    " SELECT document_id AS id FROM access WHERE mailbox = current_setting('app.person')"
    " AND from_at < %(cutoff)s),"
    " visible AS MATERIALIZED ("
    " SELECT c.id FROM readable r JOIN documents d ON d.id = r.id"
    " JOIN chunks c ON c.document_id = d.id"
    " WHERE d.sent_at >= %(since)s AND d.sent_at < %(cutoff)s)"
    " SELECT v.chunk_id AS id, 1 - (v.embedding <=> %(vector)s) AS score"
    " FROM visible x JOIN {} v ON v.chunk_id = x.id"
    " ORDER BY (v.embedding <=> %(vector)s) + 0 LIMIT %(depth)s"
)


def ranked(conn: psycopg.Connection[TupleRow], model: str, dims: int) -> sql.Composed:
    """(chunk id, cosine similarity) of the `depth` chunks nearest `vector` sent from `since` to `cutoff`."""
    person = conn.execute("SELECT current_setting('app.person', true)").fetchall()[0][0]
    query = INDEXED if person == db.INVESTIGATOR else EXACT
    return query.format(vectors_identifier(model, dims))


def search(
    conn: psycopg.Connection[TupleRow],
    query_vector: NDArray[np.float32],
    model: str,
    as_of: date,
    k: int,
) -> list[SearchHit]:
    """Top `k` chunks by cosine distance among documents sent on or before `as_of`."""
    params = {
        "vector": HalfVector(query_vector),
        "since": EARLIEST,
        "cutoff": cutoff(as_of),
        "depth": k,
    }
    return hits(conn, ranked(conn, model, len(query_vector)), params)
