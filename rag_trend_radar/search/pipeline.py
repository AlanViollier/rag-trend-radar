"""The search `ask` runs, shared with the evals so a score is always about what `ask` does."""

import time
from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.ingest.origin import Origin
from rag_trend_radar.ingest.origin import origin as origin_of
from rag_trend_radar.records import Ranked, SearchHit, SearchStage
from rag_trend_radar.search import hybrid
from rag_trend_radar.search.collapse import collapse
from rag_trend_radar.search.mix import RESULTS, mix
from rag_trend_radar.search.rerank import rerank


def reranked(
    conn: psycopg.Connection[TupleRow],
    question: str,
    query_vector: NDArray[np.float32],
    as_of: date,
    *,
    since: date | None = None,
    origin: Origin | None = None,
    model: str | None = None,
    reranker: str | None = None,
    depth: int = hybrid.DEPTH,
) -> Ranked:
    """Hybrid search's `depth` best chunks, reordered by the cross-encoder, and each stage's time.

    `since` and `origin` narrow the candidates before the rerank, which is the slow part: mail from
    that day on, chunks of that origin only. The agent narrows; `ask` and the evals don't. The
    embedding model and the reranker default to the configured ones; an experiment passes others.
    """
    settings = get_settings()
    begun = time.perf_counter()
    candidates = hybrid.search(
        conn, question, query_vector, model or settings.embedding_model, as_of, depth, since
    )
    found, fused = time.perf_counter(), len(candidates)
    if origin is not None:
        candidates = [hit for hit in candidates if origin_of(hit.text) is origin]
    ranked = rerank(conn, question, candidates, reranker or settings.rerank_model, top=depth)
    stages = (
        SearchStage(name="search", seconds=found - begun, items=fused),
        SearchStage(name="rerank", seconds=time.perf_counter() - found, items=len(candidates)),
    )
    return Ranked(hits=tuple(ranked), stages=stages)


def final(ranked: Sequence[SearchHit], depth: int = hybrid.DEPTH) -> list[SearchHit]:
    """What reaches Claude: repeats of a better result dropped, press and bulk mail capped."""
    return mix(collapse(ranked, depth), RESULTS)
