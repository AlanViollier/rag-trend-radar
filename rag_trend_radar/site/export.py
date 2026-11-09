"""The replay page's data file, from the radar tables. Reads as the investigator, writes no table."""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.days import cutoff
from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.site import names, places, smooth
from rag_trend_radar.site.records import PageAlert, PageBlob, PageData, PageEvent, PageWeek

# Weeks either side that calm a blob's size and its place, and its faster-moving numbers.
_SLOW, _QUICK = 4, 2
_NEW_BLOB_DAYS = 28
_SHAPES = {
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
    recent_press: int
    surge: float
    moving: float


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
    mailboxes: int


@dataclass(frozen=True)
class Tables:
    """Everything the page reads from the store."""

    seen: list[Seen]
    fired: list[Fired]
    days: list[Day]
    documents_before: int
    chunks_before: int


def read(conn: psycopg.Connection[TupleRow]) -> Tables:
    """The shown blobs, blobs only an alert names, the alerts, and the store day by day."""
    seen = [
        Seen(b, s, w, c.to_numpy(), tuple(k), m, r, p, u, v)
        for b, s, w, c, k, m, r, p, u, v in conn.execute(
            "SELECT b.id, b.shown, w.week, w.centre, w.keywords, w.mail, w.recent, w.recent_press,"
            " w.surge, w.moving FROM blobs b JOIN blob_weeks w ON w.blob_id = b.id"
            " WHERE b.shown OR b.id IN (SELECT blob_id FROM alerts) ORDER BY b.id, w.week"
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
            "SELECT day, documents, chunks, senders, mailboxes FROM day_totals ORDER BY day"
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
    return Tables(seen, fired, days, documents, chunks)


def events_from(path: Path) -> list[PageEvent]:
    """The real events, one JSON object a line."""
    return [PageEvent.model_validate_json(line) for line in path.read_text().splitlines()]


def page(tables: Tables, events: Sequence[PageEvent], ends: Sequence[date]) -> PageData:
    """The whole page: every shown blob smoothed and placed, its names, the alerts, the weeks."""
    where = {end: i for i, end in enumerate(ends)}
    lives: defaultdict[int, list[Seen]] = defaultdict(list)
    for row in tables.seen:
        lives[row.blob].append(row)
    shown = {blob: life for blob, life in lives.items() if life[0].shown}
    placed = _places(shown, where, len(ends))
    blobs = [
        _blob(blob, life, where, placed[row] if life[0].shown else None)
        for row, (blob, life) in enumerate(_shown_first(lives))
    ]
    return PageData(
        weeks=_weeks(tables, ends, blobs),
        blobs=tuple(blobs),
        alerts=tuple(_alert(f, where) for f in tables.fired),
        events=tuple(events),
    )


def write(data: PageData, out: Path) -> int:
    """Check the file holds nothing number-shaped it shouldn't, write it, and return its bytes."""
    text = data.model_dump_json() + "\n"
    for what, shape in _SHAPES.items():
        if shape.search(text):
            raise ExportRefused(f"the page data holds a {what} number: nothing written")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return len(text.encode())


def _shown_first(lives: Mapping[int, list[Seen]]) -> list[tuple[int, list[Seen]]]:
    return sorted(lives.items(), key=lambda item: (not item[1][0].shown, item[0]))


def _places(
    shown: Mapping[int, list[Seen]], where: Mapping[date, int], weeks: int
) -> NDArray[np.float64]:
    """Each shown blob's (x, y) every week, in the order of `_shown_first`."""
    ordered = sorted(shown)
    points = places.map_points(np.array([r.centre for b in ordered for r in shown[b]]))
    want = np.full((len(ordered), weeks, 2), np.nan)
    mail = np.zeros((len(ordered), weeks))
    used = 0
    for row, blob in enumerate(ordered):
        life = shown[blob]
        seen = [where[r.week] for r in life]
        span = np.arange(seen[0], seen[-1] + 1)
        mail[row, span] = smooth.median(smooth.fill(seen, [r.mail for r in life]), _SLOW)
        for axis in range(2):
            track = smooth.fill(seen, points[used : used + len(life), axis].tolist())
            want[row, span, axis] = smooth.median(track, _SLOW)
        used += len(life)
    return places.pack(want, mail)


def _blob(
    blob: int,
    life: Sequence[Seen],
    where: Mapping[date, int],
    placed: NDArray[np.float64] | None,
) -> PageBlob:
    seen = [where[r.week] for r in life]
    named = tuple(names.rough(seen, [r.words for r in life]))
    if placed is None:
        return PageBlob(id=blob, first=seen[0], shown=False, names=named)

    def calm(values: Sequence[float], weeks: int) -> NDArray[np.float64]:
        return smooth.median(smooth.fill(seen, values), weeks)

    recent = calm([r.recent for r in life], _QUICK)
    press = calm([r.recent_press for r in life], _QUICK)
    span = np.arange(seen[0], seen[-1] + 1)
    return PageBlob(
        id=blob,
        first=seen[0],
        shown=True,
        names=named,
        mail=_whole(calm([r.mail for r in life], _SLOW)),
        recent=_whole(recent),
        surge=_whole(calm([r.surge for r in life], _QUICK)),
        moving=_rounded(calm([r.moving for r in life], _QUICK), 4),
        press=_rounded(np.clip(press / np.maximum(recent, 1), 0, 1), 2),
        x=_rounded(placed[span, 0], 4),
        y=_rounded(placed[span, 1], 4),
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
                mailboxes=max((d.mailboxes for d in this_week), default=0),
                senders=max((d.senders for d in this_week), default=0),
                live_blobs=sum(b.shown and b.first <= i < b.first + len(b.mail) for b in blobs),
            )
        )
    return tuple(weeks)


def _alert(fired: Fired, where: Mapping[date, int]) -> PageAlert:
    return PageAlert(
        day=fired.day,
        week=where[fired.day],
        blob=fired.blob,
        observed=round(fired.observed),
        normal=round(fired.expected, 1),
        kind="new" if (fired.day - fired.born).days < _NEW_BLOB_DAYS else "speeding",
    )


def _whole(values: NDArray[np.float64]) -> tuple[int, ...]:
    return tuple(np.rint(values).astype(int).tolist())


def _rounded(values: NDArray[np.float64], digits: int) -> tuple[float, ...]:
    return tuple(np.round(values, digits).tolist())
