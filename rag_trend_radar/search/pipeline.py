"""The search `ask` runs, shared with the evals so a score is always about what `ask` does."""

import time
from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.ingest.origin import Origin
from rag_trend_radar.ingest.origin import origin as origin_of
from rag_trend_radar.records import SearchHit, SearchStage
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
    origin: Origin | None = None,
    stages: list[SearchStage] | None = None,
) -> list[SearchHit]:
    """Hybrid search's `depth` best chunks, reordered by the cross-encoder.

    `since` and `origin` narrow the candidates before the rerank, which is the slow part: mail from
    that day on, chunks of that origin only. The agent narrows; `ask` and the evals don't.
    `stages`, when given, gets the search's and the rerank's times appended (page 2 shows them).
    """
    begun = time.perf_counter()
    candidates = hybrid.search(conn, question, query_vector, model, as_of, depth, since)
    found, fused = time.perf_counter(), len(candidates)
    if origin is not None:
        candidates = [hit for hit in candidates if origin_of(hit.text) is origin]
    ranked = rerank(question, candidates, reranker, top=depth)
    if stages is not None:
        reranking = time.perf_counter() - found
        stages.append(SearchStage(name="search", seconds=found - begun, items=fused))
        stages.append(SearchStage(name="rerank", seconds=reranking, items=len(candidates)))
    return ranked


def final(ranked: Sequence[SearchHit], depth: int) -> list[SearchHit]:
    """What reaches Claude: repeats of a better result dropped, press and bulk mail capped."""
    return mix(collapse(ranked, depth), RESULTS)
