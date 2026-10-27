"""recall@k and friends, computed by ranx so they match trec_eval."""

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from ranx import Qrels, Run, evaluate

from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import EvalQuestion, SearchHit
from rag_trend_radar.search import hybrid, keyword, vector
from rag_trend_radar.search.rerank import rerank


def score(
    relevant: Mapping[str, Collection[int]],
    results: Mapping[str, Sequence[SearchHit]],
    metrics: Sequence[str],
) -> dict[str, float]:
    """Score each question's ranked documents against its relevant ones, averaged over questions.

    Results are chunks; a document ranks where its best chunk does. Scores handed to ranx are
    1 / rank, so methods with incomparable raw scores (BM25, cosine, RRF) are scored alike.
    """
    run: dict[str, dict[str, float]] = {}
    for qid in relevant:
        ranks: dict[str, float] = {}
        for hit in results.get(qid, ()):
            ranks.setdefault(str(hit.document_id), 1 / (len(ranks) + 1))
        ## ranx needs every question in the run; one that found nothing scores 0, as it should.
        run[qid] = ranks or {"none": 0.0}
    qrels = {qid: {str(d): 1 for d in docs} for qid, docs in relevant.items()}
    scores = evaluate(Qrels.from_dict(qrels), Run.from_dict(run), list(metrics))
    ## ranx returns a bare float when asked for a single metric.
    if isinstance(scores, float):
        return {metrics[0]: scores}
    return {name: float(value) for name, value in scores.items()}


def rankings(
    questions: Sequence[EvalQuestion], vectors: NDArray[np.float32], depth: int
) -> dict[str, dict[str, list[SearchHit]]]:
    """Every method's ranked chunks per question, each question asked as its person.

    Returns {method: {question id: hits}}. The reranker reorders the same `depth` candidates the
    method under it found, so its row shows what reordering adds and nothing else.
    """
    settings = get_settings()
    model, reranker = settings.embedding_model, settings.rerank_model
    out: dict[str, dict[str, list[SearchHit]]] = defaultdict(dict)
    for q, vector_ in zip(questions, vectors, strict=True):
        with db.connect(q.person) as conn:
            lists = {
                "keyword": keyword.search(conn, q.question, q.as_of, depth),
                "vector": vector.search(conn, vector_, model, q.as_of, depth),
                "hybrid": hybrid.search(conn, q.question, vector_, model, q.as_of, depth),
            }
        lists["vector + rerank"] = rerank(q.question, lists["vector"], reranker, depth)
        lists["hybrid + rerank"] = rerank(q.question, lists["hybrid"], reranker, depth)
        for method, hits in lists.items():
            out[method][q.id] = hits
    return dict(out)
