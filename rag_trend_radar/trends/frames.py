"""One map and one set of blobs per week over the last 26 weeks of mail, each map started from the last.

Saved as files: a map takes a minute or two, and everything else is recomputed from them.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.run_log import RunLog
from rag_trend_radar.trends import topics

## Locked with the owner 2026-11-05. 26 weeks: 8 and 16 kept a blob too briefly (median life, mail in
## lasting blobs). First window Jan–Jul 2000: too little mail before. After Feb 2002 mailboxes empty out.
WEEKS, FIRST, LAST = 26, date(2000, 7, 2), date(2002, 2, 24)
SEEDS = (1, 2)
## Bump by hand when the clustering changes: a code hash would re-cluster 174 frames, about 3 hours,
## on any edit to a docstring.
CLUSTER_VERSION = "v1"
_NEAREST = 5


@dataclass(frozen=True)
class Frame:
    """One week: which pool rows its window holds, and each row's blob."""

    end: date
    rows: NDArray[np.int64]
    labels: NDArray[np.int64]


def week_ends(first: date = FIRST, last: date = LAST) -> list[date]:
    """Every week's last day from `first` to `last`."""
    return [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]


def window(chosen: topics.Pool, end: date) -> NDArray[np.int64]:
    """The pool rows sent in the 26 weeks up to `end`."""
    days = np.array([d.toordinal() for d in chosen.days])
    start = (end - timedelta(weeks=WEEKS)).toordinal()
    return np.flatnonzero((days > start) & (days <= end.toordinal()))


def make(
    chosen: topics.Pool,
    ends: Sequence[date],
    seed: int,
    folder: Path,
    log: RunLog,
    min_size: int = topics.MIN_SIZE,
) -> None:
    """Fill `folder` with each week's map and blobs; a week already there is kept, so it resumes."""
    folder.mkdir(parents=True, exist_ok=True)
    previous: tuple[NDArray[np.int64], NDArray[np.float32]] | None = None
    for end in ends:
        rows = window(chosen, end)
        map_file = folder / f"{end}.npy"
        if map_file.exists():
            points = np.load(map_file)
        else:
            with log.stage("map") as handled:
                start = None if previous is None else warm_start(chosen.vectors, rows, *previous)
                neighbors = min(topics.NEIGHBORS, len(rows) - 1)
                points = topics.squash(chosen.vectors[rows], topics.SQUASH, neighbors, seed, start)
                handled.items = len(rows)
            np.save(map_file, points)
        previous = rows, points
        labels_file = _labels_file(folder, end)
        if not labels_file.exists():
            with log.stage("blobs") as handled:
                labels = topics.drop_thin(
                    topics.cluster(points, min_size), chosen.document_ids[rows]
                )
                labels, _ = topics.drop_loose(labels, chosen.vectors[rows])
                handled.items = len(rows)
            np.save(labels_file, labels)


def load(chosen: topics.Pool, ends: Sequence[date], folder: Path) -> list[Frame]:
    """The saved weeks, row for row with `chosen`."""
    return [Frame(end, window(chosen, end), np.load(_labels_file(folder, end))) for end in ends]


def warm_start(
    vectors: NDArray[np.float32],
    rows: NDArray[np.int64],
    old_rows: NDArray[np.int64],
    old_points: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Where each chunk begins: where it was, or for new mail, among its 5 nearest old chunks."""
    place = np.full(len(vectors), -1, dtype=np.int64)
    place[old_rows] = np.arange(len(old_rows))
    start = np.zeros((len(rows), old_points.shape[1]), dtype=np.float32)
    known = place[rows] != -1
    start[known] = old_points[place[rows[known]]]
    new = np.flatnonzero(~known)
    old_vectors = vectors[old_rows]
    for i in range(0, len(new), 2000):
        part = new[i : i + 2000]
        similar = vectors[rows[part]] @ old_vectors.T
        nearest = np.argpartition(-similar, _NEAREST, axis=1)[:, :_NEAREST]
        start[part] = old_points[nearest].mean(axis=1)
    return start


def _labels_file(folder: Path, end: date) -> Path:
    return folder / f"{end}-labels-{CLUSTER_VERSION}.npy"
