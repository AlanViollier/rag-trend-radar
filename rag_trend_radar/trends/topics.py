"""Find topics: the chunks a period's clustering runs on, the clusters, and how good they look."""

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
from rag_trend_radar.trends.words import keywords, reading_text

NOISE = -1
## Picked on Jan–Jun 2000 (2026-11-03): 157 topics, 41% unassigned, against 89% and near-copies
## with no squashing.
SQUASH, NEIGHBORS, MIN_SIZE, MIN_SAMPLES = 5, 15, 30, 5
MIN_DOCUMENTS = 5
## Topics from Jan–Jun 2000 start at about .86 (5th percentile); the one that swallowed the mail, .83.
MIN_COHESION = 0.84


@dataclass(frozen=True)
class Pool:
    """Chunks to cluster, row for row: who they are and their vectors."""

    chunk_ids: NDArray[np.int64]
    document_ids: NDArray[np.int64]
    texts: list[str]
    vectors: NDArray[np.float32]
    days: list[date]
    press: list[bool]

    def take(self, rows: NDArray[np.int64]) -> Pool:
        """The same pool cut down to `rows`."""
        return Pool(
            self.chunk_ids[rows],
            self.document_ids[rows],
            [self.texts[i] for i in rows],
            self.vectors[rows],
            [self.days[i] for i in rows],
            [self.press[i] for i in rows],
        )


@dataclass(frozen=True)
class Found:
    """One cluster, ready to read: its size, its distinctive words, the mail nearest its centre."""

    label: int
    chunks: int
    documents: int
    cohesion: float
    keywords: tuple[str, ...]
    nearest: tuple[str, ...]


def pool(
    conn: psycopg.Connection[TupleRow],
    model: str,
    dims: int,
    since: date,
    until: date,
    *,
    unassigned_only: bool = False,
) -> Pool:
    """Chunks sent from `since` to `until` that say something: no short replies, no bulk mail.

    A short reply's vector is mostly the message it answered, and fifteen "ok"s aren't a topic.
    `unassigned_only` keeps the chunks no topic has taken yet.
    """
    waiting = "AND NOT EXISTS (SELECT 1 FROM assignments a WHERE a.chunk_id = c.id)"
    rows = conn.execute(
        sql.SQL(
            "SELECT c.id, c.document_id, c.text, v.embedding,"
            " (d.sent_at AT TIME ZONE %s)::date FROM chunks c"
            " JOIN documents d ON d.id = c.document_id JOIN {} v ON v.chunk_id = c.id"
            " WHERE d.deleted_at IS NULL AND d.sent_at >= %s AND d.sent_at < %s"
            " AND c.quote_chars = 0 {} ORDER BY c.id"
        ).format(
            index.vectors_identifier(model, dims),
            sql.SQL(waiting if unassigned_only else ""),
        ),
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
    )


def shorten(vectors: NDArray[np.float32], dims: int) -> NDArray[np.float32]:
    """The first `dims` numbers of each vector, back to length 1: EmbeddingGemma is built for it."""
    short = vectors[:, :dims]
    return (short / np.linalg.norm(short, axis=1, keepdims=True)).astype(np.float32)


def find(
    vectors: NDArray[np.float32],
    document_ids: NDArray[np.int64],
    min_size: int = MIN_SIZE,
    seed: int = 1,
) -> tuple[NDArray[np.int64], int]:
    """Topics in some chunks, the way the radar finds them.

    Returns a cluster number per row, and how many clusters were too loose to be a topic.
    """
    if len(vectors) < min_size:
        return np.full(len(vectors), NOISE, dtype=np.int64), 0
    points = squash(vectors, SQUASH, min(NEIGHBORS, len(vectors) - 1), seed)
    return drop_loose(drop_thin(cluster(points, min_size, MIN_SAMPLES), document_ids), vectors)


def drop_loose(
    labels: NDArray[np.int64], vectors: NDArray[np.float32], min_cohesion: float = MIN_COHESION
) -> tuple[NDArray[np.int64], int]:
    """Clusters whose members barely resemble their centre go back to `NOISE`; how many did.

    Such a centre is the average of unrelated mail, which sits near all mail, so everything joins
    it: one born at .83 took 31,591 chunks in six months (2026-11-03).
    """
    kept = labels.copy()
    dropped = 0
    for number in set(labels.tolist()) - {NOISE}:
        rows = labels == number
        centre = vectors[rows].mean(axis=0)
        centre /= np.linalg.norm(centre)
        if float((vectors[rows] @ centre).mean()) < min_cohesion:
            kept[rows] = NOISE
            dropped += 1
    return kept, dropped


def squash(
    vectors: NDArray[np.float32], components: int, neighbors: int, seed: int
) -> NDArray[np.float32]:
    """UMAP: the same chunks in `components` numbers each, similar ones kept close.

    In 768 numbers every chunk sits about as far from every other, so density-based clustering
    only finds near-copies. A fixed seed makes it reproducible and single-threaded.
    """
    return np.asarray(
        UMAP(
            n_components=components,
            n_neighbors=neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        ).fit_transform(vectors),
        dtype=np.float32,
    )


def cluster(
    vectors: NDArray[np.float32], min_size: int, min_samples: int, selection: str = "eom"
) -> NDArray[np.int64]:
    """A cluster number per row, `NOISE` for a row in none.

    On length-1 vectors straight-line distance orders pairs as cosine does, and the trees that keep
    memory flat need a metric like it: pairwise cosine would hold every pair, ~5 GB at 25k chunks.
    """
    model = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=selection,
        n_jobs=-1,
        copy=True,
    )
    return np.asarray(model.fit_predict(vectors), dtype=np.int64)


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


def describe(
    chosen: Pool, vectors: NDArray[np.float32], labels: NDArray[np.int64], subjects: int = 5
) -> list[Found]:
    """Every cluster's words, cohesion and nearest mail, biggest first."""
    numbers = sorted(set(labels.tolist()) - {NOISE})
    texts = [
        " ".join(reading_text(chosen.texts[i]) for i in np.flatnonzero(labels == n))
        for n in numbers
    ]
    words = keywords(texts)
    found: list[Found] = []
    for number, top in zip(numbers, words, strict=True):
        rows = np.flatnonzero(labels == number)
        centre = vectors[rows].mean(axis=0)
        centre /= np.linalg.norm(centre)
        similarity = vectors[rows] @ centre
        closest = rows[np.argsort(-similarity)[: subjects * 3]]
        found.append(
            Found(
                label=number,
                chunks=len(rows),
                documents=len(set(chosen.document_ids[rows].tolist())),
                cohesion=float(similarity.mean()),
                keywords=tuple(top),
                nearest=tuple(dict.fromkeys(_subject(chosen.texts[i]) for i in closest))[:subjects],
            )
        )
    return sorted(found, key=lambda f: -f.documents)


def _subject(text: str) -> str:
    first = text.partition("\n")[0]
    return first.removeprefix("Subject: ") if first.startswith("Subject: ") else "(no subject)"
