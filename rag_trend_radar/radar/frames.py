"""One map and one set of clusters per week over the last 26 weeks of mail, each map started from the last.

Saved as files: a map takes a minute or two, and everything else is recomputed from them. Each week
keeps the chunk ids it was made from, so a chunk deleted later just drops out of its cluster. Those ids
are only good in the store that handed them out, so each folder records that store and refuses another.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.exceptions import OtherStore
from rag_trend_radar.radar import clustering
from rag_trend_radar.radar.clustering import NOISE
from rag_trend_radar.run_log import RunLog

# Locked with the owner 2026-11-05. 26 weeks: 8 and 16 kept a cluster too briefly (median life, mail in
# lasting clusters). First window Jan–Jul 2000: too little mail before. After Feb 2002 mailboxes empty out.
WEEKS, FIRST, LAST = 26, date(2000, 7, 2), date(2002, 2, 24)
SEEDS = (1, 2)
# Bump by hand when the clustering changes: a code hash would re-cluster 174 frames, about 3 hours,
# on any edit to a docstring.
CLUSTER_VERSION = "v1"
_NEAREST = 5
STAMP = "store.txt"


@dataclass(frozen=True)
class Frame:
    """One week: which pool rows its window holds, and each row's cluster."""

    end: date
    rows: NDArray[np.int64]
    labels: NDArray[np.int64]


def week_ends(first: date = FIRST, last: date = LAST) -> list[date]:
    """Every week's last day from `first` to `last`."""
    return [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]


def window(pool: clustering.Pool, end: date) -> NDArray[np.int64]:
    """The pool rows sent in the 26 weeks up to `end`."""
    days = np.array([d.toordinal() for d in pool.days])
    start = (end - timedelta(weeks=WEEKS)).toordinal()
    return np.flatnonzero((days > start) & (days <= end.toordinal()))


def make(
    pool: clustering.Pool,
    ends: Sequence[date],
    seed: int,
    folder: Path,
    log: RunLog,
    min_size: int = clustering.MIN_SIZE,
) -> None:
    """Fill `folder` with each week's map and clusters; a week already there is kept, so it resumes."""
    folder.mkdir(parents=True, exist_ok=True)
    if not any(folder.iterdir()):
        (folder / STAMP).write_text(pool.store)
    _same_store(folder, pool.store)
    previous: tuple[NDArray[np.int64], NDArray[np.float32]] | None = None
    for end in ends:
        map_file = folder / f"{end}.npy"
        if map_file.exists():
            rows, kept = _matched(pool, folder, end)
            points = np.load(map_file)[kept]
        else:
            rows = window(pool, end)
            kept = np.ones(len(rows), dtype=bool)
            with log.stage("map") as handled:
                start = None if previous is None else warm_start(pool.vectors, rows, *previous)
                neighbors = min(clustering.NEIGHBORS, len(rows) - 1)
                points = clustering.squash(
                    pool.vectors[rows], clustering.SQUASH, neighbors, seed, start
                )
                handled.items = len(rows)
            np.save(map_file, points)
            np.save(_chunks_file(folder, end), pool.chunk_ids[rows])
        previous = rows, points
        labels_file = _labels_file(folder, end)
        if not labels_file.exists():
            with log.stage("clusters") as handled:
                labels = clustering.drop_thin(
                    clustering.cluster(points, min_size), pool.document_ids[rows]
                )
                labels = clustering.drop_loose(labels, pool.vectors[rows])
                handled.items = len(rows)
            # Saved row for row with the chunk ids, deleted chunks as noise.
            full = np.full(len(kept), NOISE, dtype=np.int64)
            full[kept] = labels
            np.save(labels_file, full)


def load(pool: clustering.Pool, ends: Sequence[date], folder: Path) -> list[Frame]:
    """The saved weeks matched to `pool` by chunk id: a chunk deleted since is in no week."""
    _same_store(folder, pool.store)
    found = []
    for end in ends:
        rows, kept = _matched(pool, folder, end)
        found.append(Frame(end, rows, np.load(_labels_file(folder, end))[kept]))
    return found


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


def _matched(
    pool: clustering.Pool, folder: Path, end: date
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Where each chunk the week was made from sits in `pool` now, and which are still there."""
    saved = np.load(_chunks_file(folder, end))
    # The pool comes ordered by chunk id.
    at = np.minimum(np.searchsorted(pool.chunk_ids, saved), len(pool.chunk_ids) - 1)
    kept: NDArray[np.bool_] = pool.chunk_ids[at] == saved
    rows: NDArray[np.int64] = at[kept]
    return rows, kept


def _same_store(folder: Path, store: str) -> None:
    stamp = folder / STAMP
    saved = stamp.read_text() if stamp.exists() else "an unknown store"
    if saved != store:
        raise OtherStore(
            f"{folder} was made from {saved}, the store is now {store}: move it aside and run"
            " radar-frames (about 2 hours)"
        )


def _labels_file(folder: Path, end: date) -> Path:
    return folder / f"{end}-labels-{CLUSTER_VERSION}.npy"


def _chunks_file(folder: Path, end: date) -> Path:
    return folder / f"{end}-chunks.npy"
