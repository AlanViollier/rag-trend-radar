"""Where each blob sits: one map for the whole run, then pushed apart week by week so bubbles don't overlap."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from umap import UMAP

from rag_trend_radar.site import smooth

# A bubble's radius, as a share of the map's side: area follows the mail.
RADIUS_PER_ROOT, RADIUS_BASE = 0.00085, 0.002
# 60 neighbours keeps a blob's weeks together, and the map's large shape over the whole run.
_NEIGHBOURS, _MIN_DIST, _SEED = 60, 0.6, 1
_MARGIN = 0.06
# Each week starts mostly where the last ended, so a blob drifts instead of jumping.
_KEEP = 0.925
_ROOM, _PUSHES, _PUSH, _PULL = 1.25, 50, 0.25, 0.01
_EDGE = 0.02
_SETTLE_WEEKS = 3


def radius(mail: NDArray[np.float64]) -> NDArray[np.float64]:
    """A bubble's radius for its mail."""
    return RADIUS_PER_ROOT * np.sqrt(mail) + RADIUS_BASE


def map_points(centres: NDArray[np.float32]) -> NDArray[np.float64]:
    """Every blob-week's centre as a point in the unit square; the same input gives the same map."""
    reducer = UMAP(
        n_components=2,
        n_neighbors=_NEIGHBOURS,
        min_dist=_MIN_DIST,
        metric="cosine",
        random_state=_SEED,
    )
    points = cast("NDArray[np.float64]", reducer.fit_transform(centres))
    spread = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    return spread * (1 - 2 * _MARGIN) + _MARGIN


def pack(want: NDArray[np.float64], mail: NDArray[np.float64]) -> NDArray[np.float64]:
    """Where each blob (row) goes each week (column), given where it wants to be and its mail.

    NaN in `want` means the blob isn't alive that week, and stays NaN.
    """
    alive = ~np.isnan(want[:, :, 0])
    placed = np.full_like(want, np.nan)
    for week in range(want.shape[1]):
        on = np.flatnonzero(alive[:, week])
        if len(on) == 0:
            continue
        goal = want[on, week]
        start = goal.copy()
        if week > 0:
            had = alive[on, week - 1]
            start[had] = placed[on[had], week - 1]
        placed[on, week] = _push_apart(
            _KEEP * start + (1 - _KEEP) * goal, goal, radius(mail[on, week]) * _ROOM
        )
    for row, living in enumerate(alive):
        weeks = np.flatnonzero(living)
        for axis in range(2):
            placed[row, weeks, axis] = smooth.mean(placed[row, weeks, axis], _SETTLE_WEEKS)
    return placed


def _push_apart(
    points: NDArray[np.float64], goal: NDArray[np.float64], radii: NDArray[np.float64]
) -> NDArray[np.float64]:
    for _ in range(_PUSHES):
        between = points[:, None] - points[None]
        distance = np.linalg.norm(between, axis=2) + 1e-6
        np.fill_diagonal(distance, 9)
        overlap = np.maximum(radii[:, None] + radii[None] - distance, 0)
        push = (between / distance[..., None] * overlap[..., None]).sum(axis=1)
        points = np.clip(points + push * _PUSH + (goal - points) * _PULL, _EDGE, 1 - _EDGE)
    return points
