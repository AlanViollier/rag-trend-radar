"""The search `ask` runs, shared with the evals so a score is always about what `ask` does."""

from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search import hybrid
from rag_trend_radar.search.collapse import collapse
from rag_trend_radar.search.mix import RESULTS, mix
from rag_trend_radar.search.rerank import rerank


def reranked(
    conn: psycopg.Connection[TupleRow],
    question: str,
    query_vector: NDArray[np.float32],
    model: str,
    as_of: date,
    reranker: str,
    depth: int,
) -> list[SearchHit]:
    """Hybrid search's `depth` best chunks, reordered by the cross-encoder."""
    candidates = hybrid.search(conn, question, query_vector, model, as_of, depth)
    return rerank(question, candidates, reranker, top=depth)


def final(ranked: Sequence[SearchHit], depth: int) -> list[SearchHit]:
    """What reaches Claude: repeats of a better result dropped, press and bulk mail capped."""
    return mix(collapse(ranked, depth), RESULTS)
