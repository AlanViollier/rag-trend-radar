"""The search `ask` runs, shared with the evals so a score is always about what `ask` does."""

from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.ingest.kinds import Source, source
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
    since: date | None = None,
    keep: Source | None = None,
) -> list[SearchHit]:
    """Hybrid search's `depth` best chunks, reordered by the cross-encoder.

    `since` and `keep` narrow the candidates before the rerank, which is the slow part: mail from
    that day on, chunks of that source only. The agent narrows; `ask` and the evals don't.
    """
    candidates = hybrid.search(conn, question, query_vector, model, as_of, depth, since)
    if keep is not None:
        candidates = [hit for hit in candidates if source(hit.text) is keep]
    return rerank(question, candidates, reranker, top=depth)


def final(ranked: Sequence[SearchHit], depth: int) -> list[SearchHit]:
    """What reaches Claude: repeats of a better result dropped, press and bulk mail capped."""
    return mix(collapse(ranked, depth), RESULTS)
