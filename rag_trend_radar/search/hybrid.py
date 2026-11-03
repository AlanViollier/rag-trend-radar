"""Keyword and vector rankings merged by reciprocal rank fusion, in one SQL query."""

from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.days import cutoff
from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search import keyword, vector
from rag_trend_radar.search.ranking import hits

## RRF fuses ranks, not scores: BM25 scores are unbounded and cosine sits in [-1, 1], so they
## can't be added. 60 is the constant from the original paper (Cormack et al., 2009); it keeps
## rank 1 from drowning out ranks 2–10. Each list is cut at DEPTH first: deep enough that a
## chunk one list ranks 30th still counts.
RRF_K = 60
DEPTH = 50


def search(
    conn: psycopg.Connection[TupleRow],
    query: str,
    query_vector: NDArray[np.float32],
    model: str,
    as_of: date,
    k: int,
) -> list[SearchHit]:
    """Top `k` chunks by RRF: each list scores 1 / (60 + rank), summed per chunk."""
    fused = sql.SQL(
        "SELECT id, sum(1.0 / (%(rrf_k)s + rank)) AS score FROM ("
        " SELECT id, row_number() OVER (ORDER BY score DESC) AS rank FROM ({keyword}) kw"
        " UNION ALL"
        " SELECT id, row_number() OVER (ORDER BY score DESC) AS rank FROM ({vector}) vec"
        ") ranks GROUP BY id ORDER BY score DESC LIMIT %(k)s"
    ).format(keyword=keyword.RANKED, vector=vector.ranked(conn, model, len(query_vector)))
    params = {
        "query": query,
        "vector": HalfVector(query_vector),
        "cutoff": cutoff(as_of),
        "depth": DEPTH,
        "rrf_k": RRF_K,
        "k": k,
    }
    return hits(conn, fused, params)
