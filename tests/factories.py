"""Small builders the tests share."""

from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.radar.clustering import Pool
from rag_trend_radar.records import AgentRun, AgentStep, Citation, SearchHit


def take(pool: Pool, rows: NDArray[np.int64]) -> Pool:
    """The same pool cut down to `rows`: what the radar reads after those chunks are gone."""
    return Pool(
        pool.chunk_ids[rows],
        pool.document_ids[rows],
        [pool.texts[i] for i in rows],
        pool.vectors[rows],
        [pool.days[i] for i in rows],
        [pool.press[i] for i in rows],
        pool.store,
    )


def hit(chunk_id: int, text: str, score: float = 1.0) -> SearchHit:
    """A search result whose chunk and document share an id."""
    return SearchHit(chunk_id=chunk_id, document_id=chunk_id, text=text, score=score)


def unit(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    """Each vector (or the one vector) scaled to length 1, as the store keeps them."""
    scaled: NDArray[np.float32] = (
        vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    ).astype(np.float32)
    return scaled


def agent_run(
    *steps: AgentStep,
    answer: str = "a",
    found: Literal["yes", "no", "partly"] | None = "yes",
    cites: tuple[int, ...] = (),
) -> AgentRun:
    """A finished run of the investigator on the last day, with the steps and answer given."""
    return AgentRun(
        question="q",
        person="investigator",
        as_of=date(2002, 12, 31),
        model="claude-sonnet-5-5",
        prompt_version=9,
        steps=steps,
        answer=answer,
        found=found,
        citations=tuple(Citation(document_id=d, cited_text="x") for d in cites),
        cost_usd=0.1,
        seconds=10.0,
        started=datetime(2026, 11, 10, tzinfo=UTC),
        commit="c",
    )
