"""Small builders the tests share."""

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.radar.clustering import Pool


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
