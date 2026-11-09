"""The data inside the page. A blob's lists run week by week from its `first` week."""

from datetime import date
from typing import Literal

from rag_trend_radar.schema import Record


class PageWeek(Record):
    """One week of the replay. `documents` and `chunks` are the store so far; `senders` the most
    different people who sent mail on one day of the week."""

    day: date
    documents: int
    chunks: int
    mail: int
    senders: int
    live_blobs: int


class PageBlob(Record):
    """A blob smoothed for the picture, later mail allowed. `shown` false: only in the full view."""

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
    """An alert as it fired. `new` is a blob under 4 weeks old, whose normal is close to nothing."""

    day: date
    week: int
    blob: int
    observed: int
    normal: float
    kind: Literal["new", "speeding"]
    note: str = ""
    weight: str = ""


class PageEvent(Record):
    """Something that really happened, on its date."""

    id: str
    day: date
    title: str


class Moment(Record):
    """Where the replay stops to explain an alert. `text` may hold {observed}, {normal}, {mail}."""

    day: date
    blob: int
    text: str


class PageMoment(Moment):
    """A moment with its numbers filled in, and the week it stops on."""

    week: int


class PageData(Record):
    """Everything the replay draws."""

    weeks: tuple[PageWeek, ...]
    blobs: tuple[PageBlob, ...]
    alerts: tuple[PageAlert, ...]
    events: tuple[PageEvent, ...]
    moments: tuple[PageMoment, ...]
