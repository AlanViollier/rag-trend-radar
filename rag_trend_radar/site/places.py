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
# How a lifted map begins: the flat one at UMAP's own scale, with some depth to grow from.
_START_SCALE, _START_DEPTH = 10.0, 1.0
_MARGIN = 0.06
# Each week starts mostly where the last ended, so a blob drifts instead of jumping.
_KEEP = 0.925
_ROOM, _PUSHES, _PUSH, _PULL = 1.25, 50, 0.25, 0.01
_EDGE = 0.02
_SETTLE_WEEKS = 3


def radius(mail: NDArray[np.float64]) -> NDArray[np.float64]:
    """A bubble's radius for its mail."""
    return RADIUS_PER_ROOT * np.sqrt(mail) + RADIUS_BASE


def map_points(
    main: NDArray[np.float32], others: NDArray[np.float32]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Blob-week centres as points in the unit square: `main` shapes the map, `others` join it."""
    reducer = UMAP(
        n_components=2,
        n_neighbors=_NEIGHBOURS,
        min_dist=_MIN_DIST,
        metric="cosine",
        random_state=_SEED,
    )
    return _embedded(reducer, main, others)


def lift(
    main: NDArray[np.float32], flat: NDArray[np.float64], others: NDArray[np.float32]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The same blob-weeks in three dimensions, started from the flat map so neighbours stay put."""
    depth = np.random.default_rng(_SEED).normal(0, _START_DEPTH, len(flat))
    reducer = UMAP(
        n_components=3,
        n_neighbors=_NEIGHBOURS,
        min_dist=_MIN_DIST,
        metric="cosine",
        random_state=_SEED,
        init=np.column_stack([flat * _START_SCALE, depth]),
    )
    return _embedded(reducer, main, others)


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
        for axis in range(want.shape[2]):
            placed[row, weeks, axis] = smooth.mean(placed[row, weeks, axis], _SETTLE_WEEKS)
    return placed


def _embedded(
    reducer: UMAP, main: NDArray[np.float32], others: NDArray[np.float32]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    points = cast("NDArray[np.float64]", reducer.fit_transform(main))
    rest = (
        cast("NDArray[np.float64]", reducer.transform(others))
        if len(others)
        else np.empty((0, points.shape[1]))
    )
    low, high = points.min(axis=0), points.max(axis=0)
    scale = (1 - 2 * _MARGIN) / (high - low)
    return (points - low) * scale + _MARGIN, (rest - low) * scale + _MARGIN


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
