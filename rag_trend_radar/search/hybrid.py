"""Keyword and vector rankings merged by reciprocal rank fusion, in one SQL query."""

from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit


def search(
    conn: psycopg.Connection[TupleRow],
    query: str,
    query_vector: NDArray[np.float32],
    model: str,
    as_of: date,
    k: int,
) -> list[SearchHit]:
    """Top `k` chunks by RRF: each list scores 1 / (60 + rank), summed per chunk."""
    raise NotImplementedError
