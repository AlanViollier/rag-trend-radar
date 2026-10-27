"""Re-order a short list with a cross-encoder that reads query and chunk together."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

from rag_trend_radar.schema import SearchHit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def rerank(query: str, hits: Sequence[SearchHit], model: str, top: int) -> list[SearchHit]:
    """Score every (query, chunk) pair with `model`, keep the best `top`.

    The embedding model compared two vectors made separately; a cross-encoder reads the question
    and the chunk in one pass, so it sees which words answer which. Too slow for 17k chunks,
    fine for the 50 hybrid search returns.
    """
    if not hits:
        return []
    scores = _load(model).predict([(query, hit.text) for hit in hits])
    ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: -float(pair[0]))
    return [hit.model_copy(update={"score": float(score)}) for score, hit in ranked[:top]]


@cache
def _load(model: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415  (models extra)

    encoder: CrossEncoder = CrossEncoder(model)
    return encoder
