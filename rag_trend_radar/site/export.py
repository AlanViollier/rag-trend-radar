"""The replay page, one file with its data inside, from the radar tables. Writes no table."""

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.days import cutoff
from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.index import vectors_identifier
from rag_trend_radar.ingest.kinds import Source, source
from rag_trend_radar.schema import Record
from rag_trend_radar.site import names, places, smooth
from rag_trend_radar.site.records import (
    AlertNote,
    Moment,
    Naming,
    PageAlert,
    PageBlob,
    PageData,
    PageDust,
    PageEvent,
    PageMoment,
    PageWeek,
)

# Weeks either side that calm a blob's size and its place, and its faster-moving numbers.
_SLOW, _QUICK = 4, 2
_NEW_BLOB_DAYS = 28
# A sample of the chunks no cluster ever took, enough for a glow. Bulk mail is dropped after
# sampling, as the radar drops it, so more are read than kept.
_DUST, _DUST_READ = 4000, 6000
_DUST_WEEKS = 26
TEMPLATE = Path(__file__).with_name("page.html")
# What the published data must never hold; page 2's export checks the same.
SHAPES = {
    "SSN-shaped": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card-shaped": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


@dataclass(frozen=True)
class Seen:
    """A blob in one week, as its table row holds it."""

    blob: int
    shown: bool
    week: date
    centre: NDArray[np.float32]
    words: tuple[str, ...]
    mail: int
    recent: int
    surge: float


@dataclass(frozen=True)
class Fired:
    """An alert's row, with when its blob was born."""

    day: date
    blob: int
    observed: float
    expected: float
    born: date


@dataclass(frozen=True)
class Day:
    """One day's row of `day_totals`."""

    day: date
    documents: int
    chunks: int
    senders: int


@dataclass(frozen=True)
class Speck:
    """A chunk no cluster took, and the day it was sent."""

    vector: NDArray[np.float32]
    day: date


@dataclass(frozen=True)
class Tables:
    """Everything the page reads from the store."""

    seen: list[Seen]
    fired: list[Fired]
    days: list[Day]
    documents_before: int
    chunks_before: int
    dust: list[Speck]


def read(conn: psycopg.Connection[TupleRow], model: str, dims: int, ends: Sequence[date]) -> Tables:
    """Every blob, the alerts, and the store day by day."""
    seen = [
        Seen(b, s, w, c.to_numpy(), tuple(k), m, r, u)
        for b, s, w, c, k, m, r, u in conn.execute(
            "SELECT b.id, b.shown, w.week, w.centre, w.keywords, w.mail, w.recent, w.surge"
            " FROM blobs b JOIN blob_weeks w ON w.blob_id = b.id ORDER BY b.id, w.week"
        ).fetchall()
    ]
    fired = [
        Fired(*row)
        for row in conn.execute(
            "SELECT a.day, a.blob_id, a.observed, a.expected, b.first_week FROM alerts a"
            " JOIN blobs b ON b.id = a.blob_id ORDER BY a.day, a.blob_id"
        ).fetchall()
    ]
    days = [
        Day(*row)
        for row in conn.execute(
            "SELECT day, documents, chunks, senders FROM day_totals ORDER BY day"
        ).fetchall()
    ]
    start = cutoff(days[0].day - timedelta(days=1))
    documents = conn.execute(
        "SELECT count(*) FROM documents WHERE sent_at < %s", (start,)
    ).fetchall()[0][0]
    chunks = conn.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.sent_at < %s",
        (start,),
    ).fetchall()[0][0]
    return Tables(seen, fired, days, documents, chunks, _dust(conn, model, dims, ends))


def _dust(
    conn: psycopg.Connection[TupleRow], model: str, dims: int, ends: Sequence[date]
) -> list[Speck]:
    rows = conn.execute(
        sql.SQL(
            "SELECT v.embedding, (d.sent_at AT TIME ZONE %s)::date, c.text FROM chunks c"
            " JOIN documents d ON d.id = c.document_id JOIN {} v ON v.chunk_id = c.id"
            " WHERE d.sent_at >= %s AND d.sent_at < %s AND c.quote_chars = 0"
            " AND NOT EXISTS (SELECT 1 FROM blob_chunks b WHERE b.chunk_id = c.id)"
            " ORDER BY c.text_hash LIMIT %s"
        ).format(vectors_identifier(model, dims)),
        (
            get_settings().timezone,
            cutoff(ends[0] - timedelta(weeks=_DUST_WEEKS)),
            cutoff(ends[-1]),
            _DUST_READ,
        ),
    ).fetchall()
    return [Speck(v.to_numpy(), d) for v, d, text in rows if source(text) is not Source.BULK][
        :_DUST
    ]


def records_from[T: Record](path: Path, record: type[T]) -> list[T]:
    """A file of records, one JSON object a line."""
    return [record.model_validate_json(line) for line in path.read_text().splitlines()]


def page(
    tables: Tables,
    events: Sequence[PageEvent],
    moments: Sequence[Moment],
    ends: Sequence[date],
    named: Sequence[Naming] = (),
    read_alerts: Sequence[AlertNote] = (),
) -> PageData:
    """The whole page: every shown blob smoothed and placed, its names, the alerts, the weeks.
    What's been read by hand replaces the rough names and decides what's on the map."""
    where = {end: i for i, end in enumerate(ends)}
    reading = {n.blob: n for n in named}
    lives: defaultdict[int, list[Seen]] = defaultdict(list)
    for row in _as_read(tables.seen, named, where):
        lives[row.blob].append(row)
    ordered = _shown_first(lives)
    flat, lifted, dust_flat, dust_lifted = _places(
        [life for _, life in ordered], tables.dust, where, len(ends)
    )
    blobs = [
        _blob(blob, life, where, flat[row], lifted[row], reading.get(blob))
        for row, (blob, life) in enumerate(ordered)
    ]
    weeks = _weeks(tables, ends, blobs)
    stops = _moments(moments, tables.fired, where, weeks, events)
    notes = {(m.day, m.blob): m for m in stops}
    judged = _judged(read_alerts, tables.fired)
    return PageData(
        weeks=weeks,
        blobs=tuple(blobs),
        alerts=tuple(
            _alert(f, where, notes.get((f.day, f.blob)), judged.get((f.day, f.blob)))
            for f in tables.fired
        ),
        events=tuple(events),
        moments=stops,
        dust=_page_dust(tables.dust, ends, dust_flat, dust_lifted),
    )


def write(data: PageData, out: Path) -> int:
    """Put the data into the page, unless it holds a number it shouldn't; return the page's bytes."""
    text = data.model_dump_json()
    for what, shape in SHAPES.items():
        if shape.search(text):
            raise ExportRefused(f"the page data holds a {what} number: nothing written")
    html = TEMPLATE.read_text().replace("__DATA__", text.replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return len(html.encode())


def _as_read(
    seen: Sequence[Seen], named: Sequence[Naming], where: Mapping[date, int]
) -> list[Seen]:
    """Each blob on or off the map as it was read. A name that fits no blob stops the export."""
    born: dict[int, date] = {}
    for row in seen:
        born.setdefault(row.blob, row.week)
    for reading in named:
        if born.get(reading.blob) != reading.first or any(w not in where for w, _ in reading.names):
            raise ExportRefused(f"the names for blob {reading.blob} don't fit its weeks")
    topics = {n.blob: n.kind == "topic" for n in named}
    return [replace(r, shown=topics[r.blob]) if r.blob in topics else r for r in seen]


def _judged(
    read_alerts: Sequence[AlertNote], fired: Sequence[Fired]
) -> dict[tuple[date, int], AlertNote]:
    alerts = {(f.day, f.blob) for f in fired}
    for note in read_alerts:
        if (note.day, note.blob) not in alerts:
            raise ExportRefused(f"the note on {note.day} names no alert on blob {note.blob}")
    return {(n.day, n.blob): n for n in read_alerts}


def _shown_first(lives: Mapping[int, list[Seen]]) -> list[tuple[int, list[Seen]]]:
    return sorted(lives.items(), key=lambda item: (not item[1][0].shown, item[0]))


def _places(
    lives: Sequence[Sequence[Seen]],
    dust: Sequence[Speck],
    where: Mapping[date, int],
    weeks: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Each blob's flat and lifted place every week, then the dust's. Only shown blobs shape the
    map or push apart."""
    main = [life for life in lives if life[0].shown]
    others = [life for life in lives if not life[0].shown]
    dims = main[0][0].centre.size
    centres = _centres(main, dims)
    extra = np.concatenate(
        [
            _centres(others, dims),
            np.array([s.vector for s in dust], dtype=np.float32).reshape(len(dust), dims),
        ]
    )
    flat, flat_extra = places.map_points(centres, extra)
    lifted, lifted_extra = places.lift(centres, flat, extra)
    joined = len(extra) - len(dust)

    def packed(rows: Sequence[Sequence[Seen]], points: NDArray[np.float64]) -> NDArray[np.float64]:
        return places.pack(_wanted(rows, where, points, weeks), _mail(rows, where, weeks))

    return (
        np.concatenate([packed(main, flat), packed(others, flat_extra[:joined])]),
        np.concatenate([packed(main, lifted), packed(others, lifted_extra[:joined])]),
        flat_extra[joined:],
        lifted_extra[joined:],
    )


def _centres(lives: Sequence[Sequence[Seen]], dims: int) -> NDArray[np.float32]:
    centres = [r.centre for life in lives for r in life]
    return np.array(centres, dtype=np.float32).reshape(len(centres), dims)


def _page_dust(
    dust: Sequence[Speck],
    ends: Sequence[date],
    flat: NDArray[np.float64],
    lifted: NDArray[np.float64],
) -> PageDust:
    return PageDust(
        first=tuple(math.ceil((s.day - ends[0]).days / 7) for s in dust),
        x=_rounded(flat[:, 0], 3),
        y=_rounded(flat[:, 1], 3),
        lifted_x=_rounded(lifted[:, 0], 3),
        lifted_y=_rounded(lifted[:, 1], 3),
        lifted_z=_rounded(lifted[:, 2], 3),
    )


def _mail(
    lives: Sequence[Sequence[Seen]], where: Mapping[date, int], weeks: int
) -> NDArray[np.float64]:
    mail = np.zeros((len(lives), weeks))
    for row, life in enumerate(lives):
        seen = [where[r.week] for r in life]
        mail[row, seen[0] : seen[-1] + 1] = smooth.median(
            smooth.fill(seen, [r.mail for r in life]), _SLOW
        )
    return mail


def _wanted(
    ordered: Sequence[Sequence[Seen]],
    where: Mapping[date, int],
    points: NDArray[np.float64],
    weeks: int,
) -> NDArray[np.float64]:
    want = np.full((len(ordered), weeks, points.shape[1]), np.nan)
    used = 0
    for row, life in enumerate(ordered):
        seen = [where[r.week] for r in life]
        span = np.arange(seen[0], seen[-1] + 1)
        for axis in range(points.shape[1]):
            track = smooth.fill(seen, points[used : used + len(life), axis].tolist())
            want[row, span, axis] = smooth.median(track, _SLOW)
        used += len(life)
    return want


def _blob(
    blob: int,
    life: Sequence[Seen],
    where: Mapping[date, int],
    flat: NDArray[np.float64],
    lifted: NDArray[np.float64],
    reading: Naming | None,
) -> PageBlob:
    seen = [where[r.week] for r in life]
    named = (
        tuple((where[week], name) for week, name in reading.names)
        if reading
        else tuple(names.rough(seen, [r.words for r in life]))
    )

    def calm(values: Sequence[float], weeks: int) -> NDArray[np.float64]:
        return smooth.median(smooth.fill(seen, values), weeks)

    span = np.arange(seen[0], seen[-1] + 1)
    return PageBlob(
        id=blob,
        first=seen[0],
        shown=life[0].shown,
        names=named,
        family=reading.family if reading else "",
        mail=_whole(calm([r.mail for r in life], _SLOW)),
        recent=_whole(calm([r.recent for r in life], _QUICK)),
        surge=_whole(calm([r.surge for r in life], _QUICK)),
        x=_rounded(flat[span, 0], 4),
        y=_rounded(flat[span, 1], 4),
        lifted_x=_rounded(lifted[span, 0], 4),
        lifted_y=_rounded(lifted[span, 1], 4),
        lifted_z=_rounded(lifted[span, 2], 4),
    )


def _weeks(tables: Tables, ends: Sequence[date], blobs: Sequence[PageBlob]) -> tuple[PageWeek, ...]:
    documents, chunks = tables.documents_before, tables.chunks_before
    days = iter(tables.days)
    day = next(days, None)
    weeks = []
    for i, end in enumerate(ends):
        this_week: list[Day] = []
        while day is not None and day.day <= end:
            documents += day.documents
            chunks += day.chunks
            if day.day > end - timedelta(weeks=1):
                this_week.append(day)
            day = next(days, None)
        weeks.append(
            PageWeek(
                day=end,
                documents=documents,
                chunks=chunks,
                mail=sum(d.documents for d in this_week),
                senders=max((d.senders for d in this_week), default=0),
                live_blobs=sum(b.shown and b.first <= i < b.first + len(b.mail) for b in blobs),
            )
        )
    return tuple(weeks)


def _moments(
    moments: Sequence[Moment],
    fired: Sequence[Fired],
    where: Mapping[date, int],
    weeks: Sequence[PageWeek],
    events: Sequence[PageEvent],
) -> tuple[PageMoment, ...]:
    found = {(f.day, f.blob): f for f in fired}
    happened = {e.id: e for e in events}
    stops = []
    for moment in moments:
        alert = found.get((moment.day, moment.blob))
        if alert is None:
            raise ExportRefused(f"the moment on {moment.day} names no alert on blob {moment.blob}")
        week = where[moment.day]
        text = moment.text.format(
            observed=f"{round(alert.observed):,}",
            normal=f"{round(alert.expected):,}",
            mail=f"{weeks[week].mail:,}",
        )
        follows = ""
        if moment.event:
            if moment.event not in happened:
                raise ExportRefused(f"the moment on {moment.day} follows no event {moment.event}")
            event = happened[moment.event]
            follows = f"{(moment.day - event.day).days} days after: {event.title}"
        stops.append(
            PageMoment(day=moment.day, blob=moment.blob, text=text, week=week, follows=follows)
        )
    return tuple(stops)


def _alert(
    fired: Fired, where: Mapping[date, int], moment: PageMoment | None, read: AlertNote | None
) -> PageAlert:
    note = read.note if read else ""
    weight = read.weight if read else ""
    return PageAlert(
        day=fired.day,
        week=where[fired.day],
        blob=fired.blob,
        observed=round(fired.observed),
        normal=round(fired.expected, 1),
        kind="new" if (fired.day - fired.born).days < _NEW_BLOB_DAYS else "speeding",
        headline=read.headline if read else "",
        note=moment.text if moment else note,
        follows=moment.follows if moment else "",
        weight="bold" if moment else weight,
    )


def _whole(values: NDArray[np.float64]) -> tuple[int, ...]:
    return tuple(np.rint(values).astype(int).tolist())


def _rounded(values: NDArray[np.float64], digits: int) -> tuple[float, ...]:
    # UMAP's transform hands back float32: rounded as it is, 0.123 prints as 0.12300000339746475.
    return tuple(np.round(values.astype(np.float64), digits).tolist())
