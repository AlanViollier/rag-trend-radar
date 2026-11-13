"""The radar's tables from the saved frames. Rows hold only what was known that week; `shown` doesn't."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.days import cutoff, opening
from rag_trend_radar.radar import alerts, cluster_kinds, links, signals
from rag_trend_radar.radar.clustering import NOISE, Pool
from rag_trend_radar.radar.frames import Frame
from rag_trend_radar.radar.words import keywords, reading_text

# Seen across fewer than 9 weekly frames, a cluster is a blip, unless the mail ends before it could prove itself.
LASTING_WEEKS = 9
ALIVE_AT_END = 2
AGREED = 0.5
# Enough to find what a cluster is about; a cluster's 7,000th chunk changes nothing, and costs memory.
KEYWORD_CHUNKS = 1500


@dataclass(frozen=True)
class ClusterWeek:
    """One cluster in one week, seeing mail up to that week only."""

    cluster: int
    week: date
    label: int
    centre: NDArray[np.float32]
    words: tuple[str, ...]
    counts: signals.Counts
    moving: float
    other_seed: bool


def cluster_weeks(pool: Pool, frames: Sequence[Frame], other: Sequence[Frame]) -> list[ClusterWeek]:
    """Every cluster's every week, oldest first, with `other` the second seed's frames."""
    lineage = links.lineages(frames)
    middle = {f.end: signals.centres(f, pool.vectors) for f in frames}
    index = {f.end: i for i, f in enumerate(frames)}
    seen: defaultdict[int, list[tuple[date, int]]] = defaultdict(list)
    for key in sorted(lineage):
        seen[lineage[key]].append(key)
    moved: dict[tuple[date, int], float] = {}
    for keys in seen.values():
        values = signals.moving([index[e] for e, _ in keys], [middle[e][n] for e, n in keys])
        moved.update(zip(keys, values, strict=True))
    found = []
    for frame, twin in zip(frames, other, strict=True):
        count = signals.counts(frame, pool, signals.counted(frame, pool.vectors))
        words = _words(frame, pool)
        agreed = set(links.same(twin, frame))
        found += [
            ClusterWeek(
                cluster=lineage[(frame.end, label)],
                week=frame.end,
                label=label,
                centre=middle[frame.end][label],
                words=tuple(words[label]),
                counts=count[label],
                moving=moved[(frame.end, label)],
                other_seed=label in agreed,
            )
            for label in sorted(count)
        ]
    return found


def write(
    conn: psycopg.Connection[TupleRow],
    pool: Pool,
    frames: Sequence[Frame],
    other: Sequence[Frame],
    model: str,
) -> dict[str, int]:
    """Replace the cluster tables and the daily totals; the same frames give the same rows."""
    weeks = cluster_weeks(pool, frames, other)
    by_cluster: defaultdict[int, list[ClusterWeek]] = defaultdict(list)
    for week in weeks:
        by_cluster[week.cluster].append(week)
    frame_of = {f.end: f for f in frames}
    index = {f.end: i for i, f in enumerate(frames)}
    mail = cluster_kinds.read(pool.texts)
    traits = {
        cluster: [
            cluster_kinds.trait(mail, _members(frame_of[w.week], w.label), w.words) for w in life
        ]
        for cluster, life in by_cluster.items()
    }
    conn.execute("TRUNCATE clusters CASCADE")
    shown = _write_clusters(conn, by_cluster, traits, index, model)
    _write_weeks(conn, weeks)
    held = _held_weeks(pool, weeks, frame_of)
    _write_held(conn, held)
    days = _write_days(conn, pool)
    everything = {f.end: _recent_everywhere(f, pool) for f in frames}
    raised = _write_alerts(conn, by_cluster, traits, index, everything)
    return {
        "clusters": len(by_cluster),
        "shown": shown,
        "cluster_weeks": len(weeks),
        "cluster_chunks": len(held),
        "days": days,
        "alerts": raised,
    }


def _write_clusters(
    conn: psycopg.Connection[TupleRow],
    by_cluster: Mapping[int, Sequence[ClusterWeek]],
    traits: Mapping[int, Sequence[cluster_kinds.Trait]],
    index: Mapping[date, int],
    model: str,
) -> int:
    shown = 0
    with conn.cursor() as cursor:
        for cluster, life in by_cluster.items():
            kind = cluster_kinds.kind(traits[cluster])
            visible = (
                kind is cluster_kinds.Kind.TOPIC
                and _lasted(index[life[0].week], index[life[-1].week], len(index))
                and float(np.mean([w.other_seed for w in life])) >= AGREED
            )
            shown += visible
            cursor.execute(
                "INSERT INTO clusters (id, model, first_week, last_week, kind, shown)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (cluster, model, life[0].week, life[-1].week, kind.value, visible),
            )
    return shown


def _write_weeks(conn: psycopg.Connection[TupleRow], weeks: Sequence[ClusterWeek]) -> None:
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO cluster_weeks (cluster_id, week, label, centre, keywords, chunks, mail, recent,"
            " recent_press, surge, moving, other_seed)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    w.cluster,
                    w.week,
                    w.label,
                    HalfVector(w.centre),
                    list(w.words),
                    w.counts.chunks,
                    w.counts.mail,
                    w.counts.recent,
                    w.counts.recent_press,
                    w.counts.surge,
                    w.moving,
                    w.other_seed,
                )
                for w in weeks
            ],
        )


def _write_held(
    conn: psycopg.Connection[TupleRow], held: Mapping[tuple[int, int], tuple[date, date]]
) -> None:
    with (
        conn.cursor() as cursor,
        cursor.copy(
            "COPY cluster_chunks (cluster_id, chunk_id, first_week, last_week) FROM STDIN"
        ) as copy,
    ):
        for (cluster, chunk), (first, last) in held.items():
            copy.write_row((cluster, chunk, first, last))


def _write_alerts(
    conn: psycopg.Connection[TupleRow],
    by_cluster: Mapping[int, Sequence[ClusterWeek]],
    traits: Mapping[int, Sequence[cluster_kinds.Trait]],
    index: Mapping[date, int],
    everything: Mapping[date, int],
) -> int:
    raised = alerts.surges(
        {
            cluster: [
                alerts.Week(index[w.week], w.week, w.counts.recent, w.counts.surge, w.other_seed, t)
                for w, t in zip(life, traits[cluster], strict=True)
            ]
            for cluster, life in by_cluster.items()
        },
        everything,
    )
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO alerts (day, cluster_id, observed, expected) VALUES (%s, %s, %s, %s)",
            [(a.day, a.cluster, a.observed, a.expected) for a in raised],
        )
    return len(raised)


def _recent_everywhere(frame: Frame, pool: Pool) -> int:
    """All the radar's chunks in the frame's last 4 weeks, in a cluster or not."""
    days = np.array([pool.days[i].toordinal() for i in frame.rows.tolist()])
    return int((days > frame.end.toordinal() - signals.RECENT_DAYS).sum())


def _lasted(first: int, last: int, frames: int) -> bool:
    return last - first + 1 >= LASTING_WEEKS or last >= frames - ALIVE_AT_END


def _members(frame: Frame, label: int) -> NDArray[np.int64]:
    members: NDArray[np.int64] = frame.rows[frame.labels == label]
    return members


def _words(frame: Frame, pool: Pool) -> dict[int, list[str]]:
    numbers = sorted(set(frame.labels.tolist()) - {NOISE})
    texts = [
        " ".join(reading_text(pool.texts[i]) for i in _members(frame, n)[:KEYWORD_CHUNKS])
        for n in numbers
    ]
    return dict(zip(numbers, keywords(texts), strict=True))


def _held_weeks(
    pool: Pool, weeks: Sequence[ClusterWeek], frame_of: dict[date, Frame]
) -> dict[tuple[int, int], tuple[date, date]]:
    """(cluster, chunk) → the first and last week the cluster held it."""
    held: dict[tuple[int, int], tuple[date, date]] = {}
    for w in weeks:
        for chunk in pool.chunk_ids[_members(frame_of[w.week], w.label)].tolist():
            first = held.get((w.cluster, chunk), (w.week, w.week))[0]
            held[(w.cluster, chunk)] = (first, w.week)
    return held


def _write_days(conn: psycopg.Connection[TupleRow], pool: Pool) -> int:
    """Each day's store and people from the tables, and what the radar saw from the pool."""
    conn.execute("DELETE FROM day_totals")
    settings = get_settings()
    first, last = min(pool.days), max(pool.days)
    since, until = opening(first), cutoff(last)
    rows = conn.execute(
        "WITH sent AS (SELECT id, kind, (sent_at AT TIME ZONE %s)::date AS day FROM documents"
        " WHERE deleted_at IS NULL AND sent_at >= %s AND sent_at < %s),"
        " docs AS (SELECT day, count(*) AS n FROM sent GROUP BY day),"
        " pieces AS (SELECT s.day, count(*) AS n FROM sent s JOIN chunks c ON c.document_id = s.id"
        " GROUP BY s.day),"
        " people AS (SELECT s.day, count(DISTINCT e.sender) AS n FROM sent s"
        " JOIN emails e ON e.document_id = s.id GROUP BY s.day),"
        " boxes AS (SELECT s.day, count(DISTINCT a.mailbox) AS n FROM sent s"
        " JOIN access a ON a.document_id = s.id WHERE s.kind = 'email' GROUP BY s.day)"
        " SELECT d.day, d.n, coalesce(p.n, 0), coalesce(e.n, 0), coalesce(b.n, 0) FROM docs d"
        " LEFT JOIN pieces p USING (day) LEFT JOIN people e USING (day)"
        " LEFT JOIN boxes b USING (day)",
        (settings.timezone, since, until),
    ).fetchall()
    seen: defaultdict[date, int] = defaultdict(int)
    press: defaultdict[date, int] = defaultdict(int)
    for day, is_press in zip(pool.days, pool.press, strict=True):
        seen[day] += 1
        press[day] += is_press
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO day_totals (day, documents, chunks, senders, mailboxes, radar_chunks,"
            " press_chunks) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(*row, seen[row[0]], press[row[0]]) for row in rows],
        )
    return len(rows)
