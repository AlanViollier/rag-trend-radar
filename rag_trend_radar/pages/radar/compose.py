"""Page 1 composed from the radar tables and what was read by hand: every cluster smoothed,
placed and named, the alerts, the weeks. Writes no table."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.pages.radar import places, rough_names, smooth
from rag_trend_radar.pages.radar.records import (
    AlertNote,
    Event,
    Moment,
    PageAlert,
    PageCluster,
    PageData,
    PageDust,
    PageMoment,
    PageWeek,
)
from rag_trend_radar.pages.radar.tables import Day, Fired, Seen, Speck, Tables
from rag_trend_radar.radar.naming import Naming

TEMPLATE = Path(__file__).with_name("page.html")
_NEW_CLUSTER_DAYS = 28


def page(
    tables: Tables,
    events: Sequence[Event],
    moments: Sequence[Moment],
    ends: Sequence[date],
    named: Sequence[Naming] = (),
    read_alerts: Sequence[AlertNote] = (),
) -> PageData:
    """The whole page: every shown cluster smoothed and placed, its names, the alerts, the weeks.
    What's been read by hand replaces the rough names and decides what's on the map."""
    where = {end: i for i, end in enumerate(ends)}
    reading = {n.cluster: n for n in named}
    lives: defaultdict[int, list[Seen]] = defaultdict(list)
    for row in _as_read(tables.seen, named, where):
        lives[row.cluster].append(row)
    ordered = _shown_first(lives)
    flat, lifted, dust_flat, dust_lifted = places.placed(
        [life for _, life in ordered], tables.dust, where, len(ends)
    )
    clusters = [
        _cluster(cluster, life, where, flat[row], lifted[row], reading.get(cluster))
        for row, (cluster, life) in enumerate(ordered)
    ]
    weeks = _weeks(tables, ends, clusters)
    stops = _moments(moments, tables.fired, where, weeks, events)
    notes = {(m.day, m.cluster): m for m in stops}
    judged = _judged(read_alerts, tables.fired)
    return PageData(
        weeks=weeks,
        clusters=tuple(clusters),
        alerts=tuple(
            _alert(f, where, notes.get((f.day, f.cluster)), judged.get((f.day, f.cluster)))
            for f in tables.fired
        ),
        events=tuple(events),
        moments=stops,
        dust=_page_dust(tables.dust, ends, dust_flat, dust_lifted),
    )


def _as_read(
    seen: Sequence[Seen], named: Sequence[Naming], where: Mapping[date, int]
) -> list[Seen]:
    """Each cluster on or off the map as it was read. A name that fits no cluster stops the export."""
    born: dict[int, date] = {}
    for row in seen:
        born.setdefault(row.cluster, row.week)
    for reading in named:
        if born.get(reading.cluster) != reading.first or any(
            w not in where for w, _ in reading.names
        ):
            raise ExportRefused(f"the names for cluster {reading.cluster} don't fit its weeks")
    topics = {n.cluster: n.kind == "topic" for n in named}
    return [replace(r, shown=topics[r.cluster]) if r.cluster in topics else r for r in seen]


def _judged(
    read_alerts: Sequence[AlertNote], fired: Sequence[Fired]
) -> dict[tuple[date, int], AlertNote]:
    alerts = {(f.day, f.cluster) for f in fired}
    for note in read_alerts:
        if (note.day, note.cluster) not in alerts:
            raise ExportRefused(f"the note on {note.day} names no alert on cluster {note.cluster}")
    return {(n.day, n.cluster): n for n in read_alerts}


def _shown_first(lives: Mapping[int, list[Seen]]) -> list[tuple[int, list[Seen]]]:
    return sorted(lives.items(), key=lambda item: (not item[1][0].shown, item[0]))


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


def _cluster(
    cluster: int,
    life: Sequence[Seen],
    where: Mapping[date, int],
    flat: NDArray[np.float64],
    lifted: NDArray[np.float64],
    reading: Naming | None,
) -> PageCluster:
    seen = [where[r.week] for r in life]
    named = (
        tuple((where[week], name) for week, name in reading.names)
        if reading
        else tuple(rough_names.rough(seen, [r.words for r in life]))
    )

    def calm(values: Sequence[float], weeks: int) -> NDArray[np.float64]:
        return smooth.median(smooth.fill(seen, values), weeks)

    span = np.arange(seen[0], seen[-1] + 1)
    return PageCluster(
        id=cluster,
        first=seen[0],
        shown=life[0].shown,
        names=named,
        family=reading.family if reading else "",
        mail=_whole(calm([r.mail for r in life], smooth.SLOW)),
        recent=_whole(calm([r.recent for r in life], smooth.QUICK)),
        surge=_whole(calm([r.surge for r in life], smooth.QUICK)),
        x=_rounded(flat[span, 0], 4),
        y=_rounded(flat[span, 1], 4),
        lifted_x=_rounded(lifted[span, 0], 4),
        lifted_y=_rounded(lifted[span, 1], 4),
        lifted_z=_rounded(lifted[span, 2], 4),
    )


def _weeks(
    tables: Tables, ends: Sequence[date], clusters: Sequence[PageCluster]
) -> tuple[PageWeek, ...]:
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
                live_clusters=sum(
                    b.shown and b.first <= i < b.first + len(b.mail) for b in clusters
                ),
            )
        )
    return tuple(weeks)


def _moments(
    moments: Sequence[Moment],
    fired: Sequence[Fired],
    where: Mapping[date, int],
    weeks: Sequence[PageWeek],
    events: Sequence[Event],
) -> tuple[PageMoment, ...]:
    found = {(f.day, f.cluster): f for f in fired}
    happened = {e.id: e for e in events}
    stops = []
    for moment in moments:
        alert = found.get((moment.day, moment.cluster))
        if alert is None:
            raise ExportRefused(
                f"the moment on {moment.day} names no alert on cluster {moment.cluster}"
            )
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
            PageMoment(
                day=moment.day, cluster=moment.cluster, text=text, week=week, follows=follows
            )
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
        cluster=fired.cluster,
        observed=round(fired.observed),
        normal=round(fired.expected, 1),
        kind="new" if (fired.day - fired.born).days < _NEW_CLUSTER_DAYS else "speeding",
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
