"""Find blobs: the chunks a period's clustering runs on, the map they're clustered on, the clusters."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg import sql
from psycopg.rows import TupleRow
from sklearn.cluster import HDBSCAN
from umap import UMAP

from rag_trend_radar import index
from rag_trend_radar.config import get_settings
from rag_trend_radar.days import cutoff
from rag_trend_radar.ingest.kinds import Source, source

NOISE = -1
## Squashing picked on Jan–Jun 2000 (89% noise and near-copies without it); 15 neighbours and size
## 100 on the weekly windows (2026-11-05): 50 neighbours moved the map more and broke a frame.
SQUASH, NEIGHBORS, MIN_SIZE, MIN_SAMPLES = 5, 15, 100, 5
MIN_DOCUMENTS = 5
## HDBSCAN can keep one blob of 95% of a frame at cohesion .837, which `drop_loose` then dumps whole.
## A cap on every frame costs 2.5 points of noise, so only a frame that would break gets one.
GIANT, CAP = 0.3, 0.05
## Topics from Jan–Jun 2000 start at about .86 (5th percentile); the one that swallowed the mail, .83.
MIN_COHESION = 0.84


@dataclass(frozen=True)
class Pool:
    """Chunks to cluster, row for row: who they are and their vectors, from the store `store`."""

    chunk_ids: NDArray[np.int64]
    document_ids: NDArray[np.int64]
    texts: list[str]
    vectors: NDArray[np.float32]
    days: list[date]
    press: list[bool]
    store: str

    def take(self, rows: NDArray[np.int64]) -> Pool:
        """The same pool cut down to `rows`."""
        return Pool(
            self.chunk_ids[rows],
            self.document_ids[rows],
            [self.texts[i] for i in rows],
            self.vectors[rows],
            [self.days[i] for i in rows],
            [self.press[i] for i in rows],
            self.store,
        )


def pool(
    conn: psycopg.Connection[TupleRow],
    model: str,
    dims: int,
    since: date,
    until: date,
) -> Pool:
    """Chunks sent from `since` to `until` that say something: no short replies, no bulk mail.

    A short reply's vector is mostly the message it answered, and fifteen "ok"s aren't a topic.
    """
    rows = conn.execute(
        sql.SQL(
            "SELECT c.id, c.document_id, c.text, v.embedding,"
            " (d.sent_at AT TIME ZONE %s)::date FROM chunks c"
            " JOIN documents d ON d.id = c.document_id JOIN {} v ON v.chunk_id = c.id"
            " WHERE d.deleted_at IS NULL AND d.sent_at >= %s AND d.sent_at < %s"
            " AND c.quote_chars = 0 ORDER BY c.id"
        ).format(index.vectors_identifier(model, dims)),
        (get_settings().timezone, cutoff(since - timedelta(days=1)), cutoff(until)),
    ).fetchall()
    kinds = [source(row[2]) for row in rows]
    kept = [(row, kind) for row, kind in zip(rows, kinds, strict=True) if kind is not Source.BULK]
    return Pool(
        np.array([r[0] for r, _ in kept], dtype=np.int64),
        np.array([r[1] for r, _ in kept], dtype=np.int64),
        [r[2] for r, _ in kept],
        np.array([r[3].to_numpy() for r, _ in kept], dtype=np.float32).reshape(len(kept), dims),
        [r[4] for r, _ in kept],
        [kind is Source.PRESS for _, kind in kept],
        str(conn.execute("SELECT id FROM store").fetchall()[0][0]),
    )


def drop_loose(
    labels: NDArray[np.int64], vectors: NDArray[np.float32], min_cohesion: float = MIN_COHESION
) -> NDArray[np.int64]:
    """Clusters whose members barely resemble their centre go back to `NOISE`.

    Such a centre is the average of unrelated mail, which sits near all mail, so everything joins
    it: one born at .83 took 31,591 chunks in six months (2026-11-03).
    """
    kept = labels.copy()
    for number in set(labels.tolist()) - {NOISE}:
        rows = labels == number
        centre = vectors[rows].mean(axis=0)
        centre /= np.linalg.norm(centre)
        if float((vectors[rows] @ centre).mean()) < min_cohesion:
            kept[rows] = NOISE
    return kept


def squash(
    vectors: NDArray[np.float32],
    components: int,
    neighbors: int,
    seed: int,
    start: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """UMAP: the same chunks in `components` numbers each, similar ones kept close.

    In 768 numbers every chunk sits about as far from every other, so density-based clustering
    only finds near-copies. A fixed seed makes it reproducible and single-threaded. `start` is where
    each chunk begins: the last map, so the new one moves 5x less for nearly the same blobs.
    """
    return np.asarray(
        UMAP(
            n_components=components,
            n_neighbors=neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
            init="spectral" if start is None else start,
        ).fit_transform(vectors),
        dtype=np.float32,
    )


def cluster(
    points: NDArray[np.float32], min_size: int = MIN_SIZE, min_samples: int = MIN_SAMPLES
) -> NDArray[np.int64]:
    """A cluster number per row, `NOISE` for a row in none; no blob swallows the frame.

    On length-1 vectors straight-line distance orders pairs as cosine does, and the trees that keep
    memory flat need a metric like it: pairwise cosine would hold every pair, ~5 GB at 25k chunks.
    """
    labels = _hdbscan(points, min_size, min_samples, None)
    placed = labels[labels != NOISE]
    if len(placed) and np.bincount(placed).max() > GIANT * len(labels):
        return _hdbscan(points, min_size, min_samples, max(2 * min_size, int(CAP * len(labels))))
    return labels


def drop_thin(
    labels: NDArray[np.int64], document_ids: NDArray[np.int64], min_documents: int = MIN_DOCUMENTS
) -> NDArray[np.int64]:
    """Clusters holding too few distinct documents go back to `NOISE`: one newsletter isn't a topic."""
    kept = labels.copy()
    for number in set(labels.tolist()) - {NOISE}:
        rows = labels == number
        if len(np.unique(document_ids[rows])) < min_documents:
            kept[rows] = NOISE
    return kept


def _hdbscan(
    points: NDArray[np.float32], min_size: int, min_samples: int, largest: int | None
) -> NDArray[np.int64]:
    model = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        max_cluster_size=largest,
        n_jobs=-1,
        copy=True,
    )
    return np.asarray(model.fit_predict(points), dtype=np.int64)
