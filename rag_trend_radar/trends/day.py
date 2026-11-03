"""The radar's replay: starting topics, then each simulated day, and each week's new topics."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg.rows import TupleRow

from rag_trend_radar.run_log import RunLog
from rag_trend_radar.schema import RadarWeek
from rag_trend_radar.trends import assign, counts, moves, topics
from rag_trend_radar.trends.moves import Centres, Registry

POOL_DAYS = 56
WEEK_ENDS_ON = 6  # Sunday


@dataclass(frozen=True)
class Setup:
    """What a replay was told: how chunks join, how centres behave, what counts as a relative."""

    centres: Centres
    join: float
    twin: float
    parent: float
    model: str
    dims: int
    min_size: int = topics.MIN_SIZE
    seed: int = 1


def replay(
    conn: psycopg.Connection[TupleRow],
    setup: Setup,
    first: date,
    until: date,
    since: date,
    log: RunLog,
) -> None:
    """Build the radar from nothing: topics from `first` up to `since`, then day by day to `until`.

    Nothing in a day looks past it: a chunk meets the centres as they stood, and births and moves
    only happen when a week closes.
    """
    with log.stage("starting topics") as handled:
        registry = start(conn, setup, first, since - timedelta(days=1))
        handled.items = len(registry.ids)
    conn.commit()
    history: list[dict[int, NDArray[np.float32]]] = []
    chunks = joined = 0
    for day in (since + timedelta(days=n) for n in range((until - since).days + 1)):
        with log.stage("one day") as handled:
            seen, took = run_day(conn, setup, registry, day)
            handled.items = seen
        chunks, joined = chunks + seen, joined + took
        if day.weekday() == WEEK_ENDS_ON:
            with log.stage("new topics") as handled:
                born = births(conn, setup, registry, day)
                handled.items = born.births
            with log.stage("centres and positions") as handled:
                earlier = (
                    history[-moves.HEADING_WEEKS] if len(history) >= moves.HEADING_WEEKS else {}
                )
                moved, positions = moves.close_week(
                    conn, day, registry, setup.centres, setup.twin, earlier, setup.model, setup.dims
                )
                handled.items = len(registry.ids)
            history.append(positions)
            log.weeks.append(
                RadarWeek(
                    day=day,
                    topics=len(registry.ids),
                    births=born.births,
                    children=born.children,
                    twins_absorbed=born.absorbed,
                    loose_clusters=born.loose,
                    chunks=chunks,
                    joined=joined,
                    moved_centres=moved.centres_moved,
                    largest_share=round(moved.largest_share, 4),
                    near_twin_pairs=moved.near_twin_pairs,
                    median_speed=round(moved.median_speed, 5),
                )
            )
            chunks = joined = 0
        conn.commit()


def start(conn: psycopg.Connection[TupleRow], setup: Setup, since: date, until: date) -> Registry:
    """The first topics, from a period's mail; what no cluster took can still join them."""
    pool = topics.pool(conn, setup.model, setup.dims, since, until)
    counts.record_totals(conn, pool)
    labels, _ = topics.find(pool.vectors, pool.document_ids, setup.min_size, setup.seed)
    numbers, centres = assign.centres_of(pool.vectors, labels)
    ids = np.array([_new_topic(conn, setup, c, until, None) for c in centres], dtype=np.int64)
    by_number = dict(zip(numbers.tolist(), ids.tolist(), strict=True))
    placed = np.flatnonzero(labels != topics.NOISE)
    _write(conn, pool, placed, np.array([by_number[labels[i]] for i in placed]), until)
    strays = np.flatnonzero(labels == topics.NOISE)
    if len(strays):
        rows, best = assign.nearest(pool.vectors[strays], centres)
        close = best >= setup.join
        _write(conn, pool, strays[close], ids[rows[close]], until)
    counts.recount(conn, pool.days)
    return Registry(ids, centres)


def run_day(
    conn: psycopg.Connection[TupleRow], setup: Setup, registry: Registry, day: date
) -> tuple[int, int]:
    """Give `day`'s chunks to the nearest centre close enough; return how many came and joined."""
    pool = topics.pool(conn, setup.model, setup.dims, day, day)
    if not len(pool.chunk_ids):
        return 0, 0
    counts.record_totals(conn, pool)
    rows, best = assign.nearest(pool.vectors, registry.centres)
    close = np.flatnonzero(best >= setup.join)
    _write(conn, pool, close, registry.ids[rows[close]], day)
    counts.recount(conn, [day])
    return len(pool.chunk_ids), len(close)


@dataclass(frozen=True)
class Born:
    """A week's new topics."""

    births: int
    children: int
    absorbed: int
    loose: int


def births(conn: psycopg.Connection[TupleRow], setup: Setup, registry: Registry, day: date) -> Born:
    """Cluster the mail no topic took in recent weeks: each cluster joins a twin or is born.

    A cluster almost touching a topic is that topic. One nearby is born with it as parent, which
    is how a split or a drift shows. Its mail is assigned today, so no earlier day knew of it.
    """
    pool = topics.pool(
        conn,
        setup.model,
        setup.dims,
        day - timedelta(days=POOL_DAYS - 1),
        day,
        unassigned_only=True,
    )
    labels, loose = topics.find(pool.vectors, pool.document_ids, setup.min_size, setup.seed)
    if not (labels != topics.NOISE).any():
        return Born(0, 0, 0, loose)
    numbers, centres = assign.centres_of(pool.vectors, labels)
    born = children = absorbed = 0
    ## Biggest first and one at a time: a cluster also meets the ones born just before it this week.
    sizes = [int((labels == number).sum()) for number in numbers.tolist()]
    for at in np.argsort(sizes, kind="stable")[::-1].tolist():
        number, centre = int(numbers[at]), centres[at]
        members = np.flatnonzero(labels == number)
        similarity = registry.centres @ centre
        row, closeness = int(similarity.argmax()), float(similarity.max())
        if closeness >= setup.twin:
            topic = int(registry.ids[row])
            absorbed += 1
        else:
            parent = int(registry.ids[row]) if closeness >= setup.parent else None
            topic = _new_topic(conn, setup, centre, day, parent)
            registry.ids = np.append(registry.ids, topic)
            registry.centres = np.vstack([registry.centres, centre])
            documents = len(np.unique(pool.document_ids[members]))
            conn.execute(
                "INSERT INTO alerts (day, kind, topic_id, observed) VALUES (%s, 'birth', %s, %s)",
                (day, topic, documents),
            )
            born += 1
            children += parent is not None
        _write(conn, pool, members, np.full(len(members), topic), day)
    counts.recount(conn, [pool.days[i] for i in np.flatnonzero(labels != topics.NOISE)])
    return Born(born, children, absorbed, loose)


def _new_topic(
    conn: psycopg.Connection[TupleRow],
    setup: Setup,
    centre: NDArray[np.float32],
    born_on: date,
    parent: int | None,
) -> int:
    row = conn.execute(
        "INSERT INTO topics (model, centre, born_on, parent_id) VALUES (%s, %s, %s, %s)"
        " RETURNING id",
        (setup.model, HalfVector(centre), born_on, parent),
    ).fetchall()
    return int(row[0][0])


def _write(
    conn: psycopg.Connection[TupleRow],
    pool: topics.Pool,
    rows: NDArray[np.int64],
    topic_ids: NDArray[np.int64],
    assigned_on: date,
) -> None:
    with (
        conn.cursor() as cur,
        cur.copy(
            "COPY assignments (chunk_id, topic_id, assigned_on, sent_on, document_id, press)"
            " FROM STDIN"
        ) as copy,
    ):
        for row, topic in zip(rows.tolist(), topic_ids.tolist(), strict=True):
            copy.write_row(
                (
                    int(pool.chunk_ids[row]),
                    topic,
                    assigned_on,
                    pool.days[row],
                    int(pool.document_ids[row]),
                    pool.press[row],
                )
            )
