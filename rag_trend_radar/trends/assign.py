"""Which topic a chunk belongs to: the nearest centre, if it's close enough."""

import numpy as np
from numpy.typing import NDArray

UNASSIGNED = -1


def centres_of(
    vectors: NDArray[np.float32], labels: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Each label's centre, length 1, in the vectors' full size; `labels` below 0 are skipped."""
    numbers = np.array(sorted(set(labels.tolist()) - {UNASSIGNED}), dtype=np.int64)
    centres = np.vstack([vectors[labels == n].mean(axis=0) for n in numbers])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return numbers, centres.astype(np.float32)


def nearest(
    vectors: NDArray[np.float32], centres: NDArray[np.float32], batch: int = 4096
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Each vector's nearest centre, as a row of `centres`, and its cosine similarity."""
    rows = np.empty(len(vectors), dtype=np.int64)
    best = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch):
        similarity = vectors[start : start + batch] @ centres.T
        rows[start : start + batch] = similarity.argmax(axis=1)
        best[start : start + batch] = similarity.max(axis=1)
    return rows, best
