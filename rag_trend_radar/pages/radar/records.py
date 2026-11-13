"""The data inside page 1, and what was written by hand for it. A cluster's lists run week by week
from its `first` week."""

from datetime import date
from pathlib import Path
from typing import Literal

from rag_trend_radar.records import Record

# Written by hand for the page: real events, the alerts read one by one, where the replay stops.
EVENTS = Path("curated/radar/events.jsonl")
ALERT_NOTES = Path("curated/radar/alert_notes.jsonl")
MOMENTS = Path("curated/radar/moments.jsonl")


class PageWeek(Record):
    """One week of the replay. `documents` and `chunks` are the store so far; `senders` the most
    different people who sent mail on one day of the week."""

    day: date
    documents: int
    chunks: int
    mail: int
    senders: int
    live_clusters: int


class PageCluster(Record):
    """A cluster smoothed for the picture, later mail allowed. `shown` false: only in the full view."""

    id: int
    first: int
    shown: bool
    names: tuple[tuple[int, str], ...]
    family: str = ""
    mail: tuple[int, ...] = ()
    recent: tuple[int, ...] = ()
    surge: tuple[int, ...] = ()
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    lifted_x: tuple[float, ...] = ()
    lifted_y: tuple[float, ...] = ()
    lifted_z: tuple[float, ...] = ()


class PageAlert(Record):
    """An alert as it fired. `new` is a cluster under 4 weeks old, whose normal is close to nothing."""

    day: date
    week: int
    cluster: int
    observed: int
    normal: float
    kind: Literal["new", "speeding"]
    headline: str = ""
    note: str = ""
    follows: str = ""
    weight: str = ""


class Event(Record):
    """Something that really happened, written down before the radar first ran: the page puts it
    on the timeline, and a moment names the one it follows."""

    id: str
    day: date
    title: str


class AlertNote(Record):
    """An alert as a person read it: bold for a real story, muted for junk, empty for routine."""

    day: date
    cluster: int
    weight: Literal["bold", "", "muted"]
    headline: str = ""
    note: str = ""


class Moment(Record):
    """Where the replay stops to explain an alert. `text` may hold {observed}, {normal}, {mail};
    `event` is the id of the real event it follows, if there's one."""

    day: date
    cluster: int
    text: str
    event: str = ""


class PageMoment(Moment):
    """A moment with its numbers filled in, the week it stops on, the event it follows."""

    week: int
    follows: str = ""


class PageDust(Record):
    """Unclustered chunks, column by column. `first` is the week each arrives, negative if early."""

    first: tuple[int, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    lifted_x: tuple[float, ...]
    lifted_y: tuple[float, ...]
    lifted_z: tuple[float, ...]


class PageData(Record):
    """Everything the replay draws."""

    weeks: tuple[PageWeek, ...]
    clusters: tuple[PageCluster, ...]
    alerts: tuple[PageAlert, ...]
    events: tuple[Event, ...]
    moments: tuple[PageMoment, ...]
    dust: PageDust
