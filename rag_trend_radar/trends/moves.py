"""Where each topic is, once a week: its centre, position, speed, and what it's moving toward."""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.index import vectors_identifier

## A position needs this many chunks to mean anything, and a centre to move.
MIN_CHUNKS = 5
POSITION_DAYS = 28
HEADING_WEEKS = 4


class Centres(StrEnum):
    """How a topic's centre behaves after birth: frozen, or recalculated weekly from its mail."""

    FROZEN = "frozen"
    ALL = "all"
    LAST_28 = "28d"
    LAST_7 = "7d"


@dataclass
class Registry:
    """The topics alive so far, row for row with the centres new mail is compared to."""

    ids: NDArray[np.int64]
    centres: NDArray[np.float32]


@dataclass(frozen=True)
class Moved:
    """A week's movement, added up."""

    centres_moved: int
    largest_share: float
    near_twin_pairs: int
    median_speed: float


def close_week(
    conn: psycopg.Connection[TupleRow],
    day: date,
    registry: Registry,
    rule: Centres,
    twin: float,
    earlier: dict[int, NDArray[np.float32]],
    model: str,
    dims: int,
) -> tuple[Moved, dict[int, NDArray[np.float32]]]:
    """Move the centres by `rule`, write every topic's week; return this week's positions.

    `earlier` holds the positions from `HEADING_WEEKS` ago, to measure speed and heading against.
    """
    table = vectors_identifier(model, dims)
    moved = 0
    if rule is not Centres.FROZEN:
        window = None if rule is Centres.ALL else (7 if rule is Centres.LAST_7 else 28)
        moved = _move(conn, day, registry, window, table)
    averages, counts = _averages(conn, day, POSITION_DAYS, table)
    positions = {n: v for n, v in averages.items() if counts[n] >= MIN_CHUNKS}
    speeds = _write(conn, day, registry, positions, counts, earlier)
    sim = registry.centres @ registry.centres.T
    total = sum(counts.values())
    return (
        Moved(
            centres_moved=moved,
            largest_share=max(counts.values(), default=0) / total if total else 0.0,
            near_twin_pairs=int((np.triu(sim, 1) >= twin).sum()),
            median_speed=float(np.median(speeds)) if speeds else 0.0,
        ),
        positions,
    )


def _averages(
    conn: psycopg.Connection[TupleRow], day: date, days: int | None, table: sql.Identifier
) -> tuple[dict[int, NDArray[np.float32]], dict[int, int]]:
    ## pgvector averages halfvecs itself, so the week never loads every member into memory.
    since = date.min if days is None else day - timedelta(days=days)
    rows = conn.execute(
        sql.SQL(
            "SELECT a.topic_id, avg(v.embedding), count(*) FROM assignments a"
            " JOIN {} v ON v.chunk_id = a.chunk_id"
            " WHERE a.sent_on > %s AND a.sent_on <= %s GROUP BY 1"
        ).format(table),
        (since, day),
    ).fetchall()
    return (
        {n: _unit(average.to_numpy()) for n, average, _ in rows},
        {n: count for n, _, count in rows},
    )


def _move(
    conn: psycopg.Connection[TupleRow],
    day: date,
    registry: Registry,
    window: int | None,
    table: sql.Identifier,
) -> int:
    averages, counts = _averages(conn, day, window, table)
    moved = 0
    for row, number in enumerate(registry.ids.tolist()):
        if counts.get(number, 0) >= MIN_CHUNKS:
            registry.centres[row] = averages[number]
            moved += 1
    return moved


def _write(
    conn: psycopg.Connection[TupleRow],
    day: date,
    registry: Registry,
    positions: dict[int, NDArray[np.float32]],
    counts: dict[int, int],
    earlier: dict[int, NDArray[np.float32]],
) -> list[float]:
    numbers = sorted(positions)
    now = np.vstack([positions[n] for n in numbers]) if numbers else np.empty((0, 1))
    similarity = now @ now.T
    np.fill_diagonal(similarity, -1.0)
    shared = [n for n in numbers if n in earlier]
    then = {n: earlier[n] for n in shared}
    speeds: list[float] = []
    rows = []
    for row, number in enumerate(registry.ids.tolist()):
        speed = nearest = closeness = heading = gain = None
        if number in positions and len(numbers) > 1:
            at = numbers.index(number)
            pick = int(similarity[at].argmax())
            nearest, closeness = numbers[pick], float(similarity[at, pick])
            if number in then:
                speed = 1.0 - float(positions[number] @ then[number])
                speeds.append(speed)
                others = [n for n in shared if n != number]
                if others:
                    gains = [
                        float(positions[number] @ positions[n] - then[number] @ then[n])
                        for n in others
                    ]
                    best = int(np.argmax(gains))
                    heading, gain = others[best], gains[best]
        rows.append(
            (
                number,
                day,
                HalfVector(registry.centres[row]),
                HalfVector(positions[number]) if number in positions else None,
                counts.get(number, 0),
                speed,
                nearest,
                closeness,
                heading,
                gain,
            )
        )
    conn.cursor().executemany(
        "INSERT INTO topic_weeks (topic_id, day, centre, position, chunks_28d, speed, nearest_id,"
        " nearest_similarity, heading_id, heading_gain)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        rows,
    )
    return speeds


def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    return (vector / np.linalg.norm(vector)).astype(np.float32)
