"""The radar's tables from the saved frames. Rows hold only what was known that week; `shown` doesn't."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.days import cutoff
from rag_trend_radar.trends import links, quiet, signals
from rag_trend_radar.trends.frames import Frame
from rag_trend_radar.trends.topics import NOISE, Pool
from rag_trend_radar.trends.words import keywords, reading_text

## Under 8 weeks a blob is a blip, unless the mail ends before it could prove itself.
LASTING_WEEKS = 9
ALIVE_AT_END = 2
AGREED = 0.5
TOP_KEYWORDS = 6
## Enough to find what a blob is about; a blob's 7,000th chunk changes nothing, and costs memory.
KEYWORD_CHUNKS = 1500


@dataclass(frozen=True)
class BlobWeek:
    """One blob in one week, seeing mail up to that week only."""

    blob: int
    week: date
    label: int
    centre: NDArray[np.float32]
    words: tuple[str, ...]
    counts: signals.Counts
    moving: float
    other_seed: bool


def blob_weeks(chosen: Pool, frames: Sequence[Frame], other: Sequence[Frame]) -> list[BlobWeek]:
    """Every blob's every week, oldest first, with `other` the second seed's frames."""
    lineage = links.lineages(frames)
    middle = {f.end: signals.centres(f, chosen.vectors) for f in frames}
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
        count = signals.counts(frame, chosen, signals.counted(frame, chosen.vectors))
        words = _words(frame, chosen)
        agreed = set(links.same(twin, frame))
        found += [
            BlobWeek(
                blob=lineage[(frame.end, label)],
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
    chosen: Pool,
    frames: Sequence[Frame],
    other: Sequence[Frame],
    model: str,
) -> dict[str, int]:
    """Replace the blob tables and the daily totals; the same frames give the same rows."""
    weeks = blob_weeks(chosen, frames, other)
    by_blob: defaultdict[int, list[BlobWeek]] = defaultdict(list)
    for week in weeks:
        by_blob[week.blob].append(week)
    conn.execute("TRUNCATE blobs CASCADE")
    frame_of = {f.end: f for f in frames}
    mail = quiet.read(chosen.texts)
    index = {f.end: i for i, f in enumerate(frames)}
    shown = 0
    with conn.cursor() as cursor:
        for blob, life in by_blob.items():
            kind = quiet.kind(
                [quiet.trait(mail, _members(frame_of[w.week], w.label), w.words) for w in life]
            )
            visible = (
                kind is quiet.Kind.TOPIC
                and _lasted(index[life[0].week], index[life[-1].week], len(frames))
                and float(np.mean([w.other_seed for w in life])) >= AGREED
            )
            shown += visible
            cursor.execute(
                "INSERT INTO blobs (id, model, first_week, last_week, kind, shown)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (blob, model, life[0].week, life[-1].week, kind.value, visible),
            )
        cursor.executemany(
            "INSERT INTO blob_weeks (blob_id, week, label, centre, keywords, chunks, mail, recent,"
            " recent_press, surge, moving, other_seed)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    w.blob,
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
    held = _held(chosen, weeks, frame_of)
    with (
        conn.cursor() as cursor,
        cursor.copy(
            "COPY blob_chunks (blob_id, chunk_id, first_week, last_week) FROM STDIN"
        ) as copy,
    ):
        for (blob, chunk), (first, last) in held.items():
            copy.write_row((blob, chunk, first, last))
    days = _write_days(conn, chosen)
    return {
        "blobs": len(by_blob),
        "shown": shown,
        "blob_weeks": len(weeks),
        "blob_chunks": len(held),
        "days": days,
    }


def _lasted(first: int, last: int, frames: int) -> bool:
    return last - first + 1 >= LASTING_WEEKS or last >= frames - ALIVE_AT_END


def _members(frame: Frame, label: int) -> NDArray[np.int64]:
    members: NDArray[np.int64] = frame.rows[frame.labels == label]
    return members


def _words(frame: Frame, chosen: Pool) -> dict[int, list[str]]:
    numbers = sorted(set(frame.labels.tolist()) - {NOISE})
    texts = [
        " ".join(reading_text(chosen.texts[i]) for i in _members(frame, n)[:KEYWORD_CHUNKS])
        for n in numbers
    ]
    return dict(zip(numbers, keywords(texts, TOP_KEYWORDS), strict=True))


def _held(
    chosen: Pool, weeks: Sequence[BlobWeek], frame_of: dict[date, Frame]
) -> dict[tuple[int, int], tuple[date, date]]:
    """(blob, chunk) → the first and last week the blob held it."""
    held: dict[tuple[int, int], tuple[date, date]] = {}
    for w in weeks:
        for chunk in chosen.chunk_ids[_members(frame_of[w.week], w.label)].tolist():
            first = held.get((w.blob, chunk), (w.week, w.week))[0]
            held[(w.blob, chunk)] = (first, w.week)
    return held


def _write_days(conn: psycopg.Connection[TupleRow], chosen: Pool) -> int:
    """Each day's store and people from the tables, and what the radar saw from the pool."""
    conn.execute("DELETE FROM day_totals")
    settings = get_settings()
    first, last = min(chosen.days), max(chosen.days)
    since, until = cutoff(date.fromordinal(first.toordinal() - 1)), cutoff(last)
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
    for day, is_press in zip(chosen.days, chosen.press, strict=True):
        seen[day] += 1
        press[day] += is_press
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO day_totals (day, documents, chunks, senders, mailboxes, radar_chunks,"
            " press_chunks) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(*row, seen[row[0]], press[row[0]]) for row in rows],
        )
    return len(rows)
