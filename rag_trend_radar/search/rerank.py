"""Re-order a short list with a cross-encoder that reads query and chunk together."""

from collections.abc import Sequence

from rag_trend_radar.schema import SearchHit


def rerank(query: str, hits: Sequence[SearchHit], model: str, top: int) -> list[SearchHit]:
    """Score every (query, chunk) pair with `model`, keep the best `top`."""
    raise NotImplementedError
