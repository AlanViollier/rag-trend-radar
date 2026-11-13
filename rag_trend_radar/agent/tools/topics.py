"""What the radar tools share: weeks in a date range, and a topic's name on a week."""

from collections.abc import Sequence
from datetime import date, timedelta
from functools import cache

from rag_trend_radar.days import week_of
from rag_trend_radar.pages.radar.export import records_from
from rag_trend_radar.pages.radar.records import NAMES, Naming


def weeks(since: date, until: date) -> tuple[date, date]:
    """The first and last radar week inside the days: a week counts once its Sunday is in range."""
    last = week_of(until)
    return week_of(since), last if last <= until else last - timedelta(weeks=1)


def name(blob: int, week: date, keywords: Sequence[str]) -> str:
    """The name the naming pass gave a blob by `week`, or its top keywords when it has none.

    Names were written reading each blob's whole life; only names dated on or before `week` show,
    so a topic isn't called what it later became.
    """
    reading = _named().get(blob)
    known = [label for start, label in reading.names if start <= week] if reading else []
    return known[-1] if known else ", ".join(keywords[:4])


def kind(blob: int, built: str) -> str:
    """Topic, newsletter, footer or chatter: the naming pass's reading, else the build's rule."""
    reading = _named().get(blob)
    return reading.kind if reading else built


def every_name(blob: int, until: date) -> list[tuple[date, str]]:
    """Every name a blob had up to `until`, with the week it started."""
    reading = _named().get(blob)
    return [(start, label) for start, label in reading.names if start <= until] if reading else []


@cache
def _named() -> dict[int, Naming]:
    return {reading.blob: reading for reading in records_from(NAMES, Naming)}
