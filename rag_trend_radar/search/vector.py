"""Nearest-meaning search over one model's vectors (pgvector HNSW)."""

from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit


def search(
    conn: psycopg.Connection[TupleRow],
    query_vector: NDArray[np.float32],
    model: str,
    as_of: date,
    k: int,
) -> list[SearchHit]:
    """Top `k` chunks by cosine distance among documents sent on or before `as_of`."""
    raise NotImplementedError
