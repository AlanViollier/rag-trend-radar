"""Re-order a short list with a cross-encoder that reads query and chunk together."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.ingest.dedup import text_hash
from rag_trend_radar.schema import SearchHit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def rerank(query: str, hits: Sequence[SearchHit], model: str, top: int) -> list[SearchHit]:
    """Score every (query, chunk) pair with `model`, keep the best `top`.

    The embedding model compared two vectors made separately; a cross-encoder reads the question
    and the chunk in one pass, so it sees which words answer which. Too slow for 17k chunks,
    fine for the 50 hybrid search returns. Scores come from `cache.rerank_scores` when this pair
    was scored before; only new pairs reach the model.
    """
    if not hits:
        return []
    revision = get_settings().revision(model)
    key = (model, revision, text_hash(query))
    texts = {text_hash(hit.text): hit.text for hit in hits}
    ## The owner's connection, on a table that holds hashes and scores, never mail: readers have
    ## no grant on it, so the agent's SQL can't plant a score.
    with db.connect() as conn:
        scores: dict[str, float] = dict(
            conn.execute(
                "SELECT text_hash, score FROM cache.rerank_scores"
                " WHERE model = %s AND revision = %s AND query_hash = %s AND text_hash = ANY(%s)",
                (*key, list(texts)),
            ).fetchall()
        )
        todo = [h for h in texts if h not in scores]
        if todo:
            predicted = _load(model, revision).predict([(query, texts[h]) for h in todo])
            fresh = {h: float(s) for h, s in zip(todo, predicted, strict=True)}
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO cache.rerank_scores VALUES (%s, %s, %s, %s, %s)"
                    " ON CONFLICT DO NOTHING",
                    [(*key, h, s) for h, s in fresh.items()],
                )
            scores |= fresh
    ranked = sorted(hits, key=lambda hit: -scores[text_hash(hit.text)])
    return [hit.model_copy(update={"score": scores[text_hash(hit.text)]}) for hit in ranked[:top]]


@cache
def _load(model: str, revision: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415  (models extra)

    encoder: CrossEncoder = CrossEncoder(model, revision=revision)
    return encoder
