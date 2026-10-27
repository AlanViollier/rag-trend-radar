"""Nearest-meaning search over one model's vectors (pgvector HNSW)."""

from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.index import vectors_identifier
from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search.ranking import cutoff, hits


def ranked(model: str, dims: int) -> sql.Composed:
    """(chunk id, cosine similarity) of the `depth` chunks nearest `vector` sent before `cutoff`."""
    return sql.SQL(
        "SELECT v.chunk_id AS id, 1 - (v.embedding <=> %(vector)s) AS score FROM {} v"
        " JOIN chunks c ON c.id = v.chunk_id JOIN documents d ON d.id = c.document_id"
        " WHERE d.sent_at < %(cutoff)s ORDER BY v.embedding <=> %(vector)s LIMIT %(depth)s"
    ).format(vectors_identifier(model, dims))


def search(
    conn: psycopg.Connection[TupleRow],
    query_vector: NDArray[np.float32],
    model: str,
    as_of: date,
    k: int,
) -> list[SearchHit]:
    """Top `k` chunks by cosine distance among documents sent on or before `as_of`."""
    params = {"vector": HalfVector(query_vector), "cutoff": cutoff(as_of), "depth": k}
    return hits(conn, ranked(model, len(query_vector)), params)
