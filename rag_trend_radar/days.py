"""Simulated days: where one ends, and which is the latest the pipeline finished."""

from datetime import date, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.exceptions import NoDaysLoaded


def cutoff(as_of: date) -> datetime:
    """The instant `as_of` ends in the store's local days: a search sees only mail sent before."""
    return datetime.combine(as_of + timedelta(days=1), time(), ZoneInfo(get_settings().timezone))


def last_day(conn: psycopg.Connection[TupleRow]) -> date:
    """The default `as_of`: the last day the pipeline finished (rule 2)."""
    day = conn.execute("SELECT max(day) FROM batches WHERE finished_at IS NOT NULL").fetchall()[0][
        0
    ]
    if day is None:
        raise NoDaysLoaded("no finished day in batches: run `rag-trend-radar replay` first")
    return cast(date, day)
