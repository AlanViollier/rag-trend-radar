"""What halfvec storage and HNSW's approximation each cost vector search, on the labelled questions.

Three rows for the configured model: exact search over float32 vectors, exact search over the
stored halfvec ones, and the HNSW search the app runs. Exact rows rank every chunk the asker
can see, listed through their own connection, so permissions match the app's.

Usage: uv run python scripts/precision.py evals/questions.jsonl evals/questions_nonames.jsonl
The float32 vectors take ~27 min on an M1 Pro the first time, then load from data/cache/.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pgvector import HalfVector
from psycopg import sql
from rich import print as show

from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed
from rag_trend_radar.evals.retrieval import score
from rag_trend_radar.index import vectors_table
from rag_trend_radar.schema import EvalQuestion, SearchHit
from rag_trend_radar.search import vector
from rag_trend_radar.search.ranking import cutoff

DEPTH = 50
METRICS = ("recall@1", "recall@10", "recall@50", "mrr@10")


def main(paths: list[Path]) -> None:
    """Print the three rows for each question file."""
    settings = get_settings()
    model, dims = settings.embedding_model, settings.embedding_dims
    with db.connect() as conn:
        rows = conn.execute("SELECT id, document_id, text FROM chunks ORDER BY id").fetchall()
        stored: dict[int, HalfVector] = dict(
            conn.execute(
                sql.SQL("SELECT chunk_id, embedding FROM {}").format(
                    sql.Identifier(vectors_table(model, dims))
                )
            ).fetchall()
        )
    ids = np.array([r[0] for r in rows])
    docs = {r[0]: r[1] for r in rows}
    halfvec = np.stack([np.array(stored[i].to_list(), dtype=np.float32) for i in ids])
    float32 = _float32(ids, [r[2] for r in rows], model, dims)

    for path in paths:
        questions = [EvalQuestion.model_validate_json(x) for x in path.read_text().splitlines()]
        relevant = _relevant(questions)
        queries = embed([q.question for q in questions], model, dims, query=True)
        results: dict[str, dict[str, list[SearchHit]]] = defaultdict(dict)
        for q, qv in zip(questions, queries, strict=True):
            with db.connect(q.person) as conn:
                visible = {
                    i
                    for (i,) in conn.execute(
                        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
                        " WHERE d.sent_at < %s",
                        (cutoff(q.as_of),),
                    ).fetchall()
                }
                results["vector, halfvec, HNSW (the app)"][q.id] = vector.search(
                    conn, qv, model, q.as_of, DEPTH
                )
            mask = np.isin(ids, list(visible))
            for name, matrix in (("float32, exact", float32), ("halfvec, exact", halfvec)):
                sims = np.where(mask, matrix @ qv, -np.inf)
                top = np.argsort(-sims)[:DEPTH]
                results[f"vector, {name}"][q.id] = [
                    SearchHit(
                        chunk_id=int(ids[j]),
                        document_id=docs[ids[j]],
                        text="",
                        score=float(sims[j]),
                    )
                    for j in top
                    if np.isfinite(sims[j])
                ]
        show(f"\n{path.name}: {len(questions)} questions, {model} @ {dims}")
        for name in (
            "vector, float32, exact",
            "vector, halfvec, exact",
            "vector, halfvec, HNSW (the app)",
        ):
            s = score(relevant, results[name], METRICS)
            show(f"  {name:34} " + "  ".join(f"{m} {s[m]:.3f}" for m in METRICS))


def _float32(ids: np.ndarray, texts: list[str], model: str, dims: int) -> np.ndarray:
    cache = get_settings().data_dir / "cache" / f"float32_{vectors_table(model, dims)}.npz"
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved["ids"], ids):
            return np.asarray(saved["vectors"], dtype=np.float32)
    vectors = embed(texts, model, dims)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=ids, vectors=vectors)
    return vectors


def _relevant(questions: list[EvalQuestion]) -> dict[str, set[int]]:
    with db.connect() as conn:
        ids: dict[str, int] = dict(
            conn.execute(
                "SELECT external_id, id FROM documents WHERE external_id = ANY(%s)",
                ([x for q in questions for x in q.relevant],),
            ).fetchall()
        )
    return {q.id: {ids[x] for x in q.relevant} for q in questions}


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
