"""recall@k and friends, computed by ranx so they match trec_eval."""

from collections.abc import Mapping, Sequence

from rag_trend_radar.schema import EvalQuestion, SearchHit


def score(
    questions: Sequence[EvalQuestion],
    results: Mapping[str, Sequence[SearchHit]],
    ks: Sequence[int],
) -> dict[str, float]:
    """Score each question's results against its relevant documents, averaged over questions."""
    raise NotImplementedError
