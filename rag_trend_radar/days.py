"""Simulated days: where one starts and ends, its radar week, and the latest the pipeline finished."""

from datetime import date, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.exceptions import NoDaysLoaded

# A search with no start date: earlier than any mail (junk dates are quarantined at staging).
EARLIEST = datetime(1900, 1, 1, tzinfo=ZoneInfo("UTC"))


def cutoff(as_of: date) -> datetime:
    """The instant `as_of` ends in the store's local days: a search sees only mail sent before."""
    return datetime.combine(as_of + timedelta(days=1), time(), ZoneInfo(get_settings().timezone))


def opening(day: date | None) -> datetime:
    """The instant `day` starts in the store's local days; with no day, before any mail."""
    return EARLIEST if day is None else cutoff(day - timedelta(days=1))


def local_day(instant: datetime) -> date:
    """The store's local day an instant falls on (America/Chicago)."""
    return instant.astimezone(ZoneInfo(get_settings().timezone)).date()


def week_of(day: date) -> date:
    """The Sunday a day's week ends on: the radar names its weeks by their last day."""
    return day + timedelta(days=6 - day.weekday())


def weeks_within(since: date, until: date) -> tuple[date, date]:
    """The first and last radar week inside the days: a week counts once its Sunday is in range."""
    last = week_of(until)
    return week_of(since), last if last <= until else last - timedelta(weeks=1)


def last_day(conn: psycopg.Connection[TupleRow]) -> date:
    """The default `as_of`: the last day the pipeline finished, never the wall clock."""
    ((day,),) = conn.execute(
        "SELECT max(day) FROM batches WHERE finished_at IS NOT NULL"
    ).fetchall()
    if day is None:
        raise NoDaysLoaded("no finished day in batches: run `rag-trend-radar replay` first")
    return cast(date, day)
