"""recall@k and friends, computed by ranx so they match trec_eval."""

import time
from collections import defaultdict
from collections.abc import Collection, Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray
from ranx import Qrels, Run, compare, evaluate
from ranx.data_structures.report import Report

from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import EvalQuestion, SearchHit
from rag_trend_radar.search import hybrid, keyword, vector
from rag_trend_radar.search.collapse import collapse
from rag_trend_radar.search.pipeline import final
from rag_trend_radar.search.rerank import rerank


def score(
    relevant: Mapping[str, Collection[Hashable]],
    ranked: Mapping[str, Sequence[Hashable]],
    metrics: Sequence[str],
) -> dict[str, float]:
    """Score each question's ranked documents against its relevant ones, averaged over questions."""
    scores = evaluate(_qrels(relevant), _run(relevant, ranked, "run"), list(metrics))
    # ranx returns a bare float when asked for a single metric.
    if isinstance(scores, float):
        return {metrics[0]: scores}
    return {name: float(value) for name, value in scores.items()}


def report(
    relevant: Mapping[str, Collection[Hashable]],
    methods: Mapping[str, Mapping[str, Sequence[Hashable]]],
    metrics: Sequence[str],
) -> Report:
    """Every method's scores in one table, a superscript where one beats another (p < 0.05).

    The test is Fisher's randomisation test, paired by question: it asks how often shuffling
    which method produced which answer would give a gap this large by chance.
    """
    runs = [_run(relevant, ranked, name) for name, ranked in methods.items()]
    return compare(_qrels(relevant), runs, list(metrics), stat_test="fisher", max_p=0.05)


def documents(hits: Sequence[SearchHit]) -> list[int]:
    """Documents in the order their best chunk ranks: a document ranks where its best chunk does."""
    return list(dict.fromkeys(hit.document_id for hit in hits))


def rankings(
    questions: Sequence[EvalQuestion],
    vectors: NDArray[np.float32],
    model: str,
    depth: int,
    reranker: str | None,
    ef_search: int = db.EF_SEARCH,
    person: str | None = None,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[float]]]:
    """Every method's ranked documents per question, asked as `person` or else the question's own.

    Returns ({method: {question id: document ids}}, {method: milliseconds per question}). The
    reranker reorders the same `depth` candidates the method under it found, so its row shows
    what reordering adds and nothing else; its time is the reordering alone, cached pairs
    included. With no `reranker`, those rows are skipped.
    """
    ranked: dict[str, dict[str, list[int]]] = defaultdict(dict)
    latency: dict[str, list[float]] = defaultdict(list)
    for q, vector_ in zip(questions, vectors, strict=True):
        lists: dict[str, list[SearchHit]] = {}
        with db.connect(person or q.person, ef_search=ef_search) as conn:
            with _clock(latency["keyword"]):
                lists["keyword"] = keyword.search(conn, q.question, q.as_of, depth)
            with _clock(latency["vector"]):
                lists["vector"] = vector.search(conn, vector_, model, q.as_of, depth)
            with _clock(latency["hybrid"]):
                lists["hybrid"] = hybrid.search(conn, q.question, vector_, model, q.as_of, depth)
        if reranker is not None:
            for under in ("vector", "hybrid"):
                with _clock(latency[f"{under} + rerank"]):
                    lists[f"{under} + rerank"] = rerank(q.question, lists[under], reranker, depth)
            with _clock(latency["hybrid + rerank + collapse"]):
                lists["hybrid + rerank + collapse"] = collapse(lists["hybrid + rerank"], depth)
            with _clock(latency["what ask uses (+ press cap)"]):
                lists["what ask uses (+ press cap)"] = final(lists["hybrid + rerank"], depth)
        for method, hits in lists.items():
            ranked[method][q.id] = documents(hits)
    return dict(ranked), dict(latency)


@contextmanager
def _clock(into: list[float]) -> Iterator[None]:
    start = time.perf_counter()
    yield
    into.append((time.perf_counter() - start) * 1000)


def _qrels(relevant: Mapping[str, Collection[Hashable]]) -> Qrels:
    return Qrels.from_dict({qid: {str(d): 1 for d in docs} for qid, docs in relevant.items()})


def _run(
    relevant: Mapping[str, Collection[Hashable]],
    ranked: Mapping[str, Sequence[Hashable]],
    name: str,
) -> Run:
    # Scores handed to ranx are 1 / rank, so methods with incomparable raw scores (BM25,
    # cosine, RRF) are scored alike. ranx needs every question in the run; one that found
    # nothing scores 0, as it should.
    run = {
        qid: {str(d): 1 / rank for rank, d in enumerate(ranked.get(qid, ()), 1)} or {"none": 0.0}
        for qid in relevant
    }
    return Run.from_dict(run, name=name)
