"""What every tool shares: who is asking as of which day, and what a tool hands back."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
from anthropic.types.beta import BetaSearchResultBlockParam, BetaTextBlockParam
from psycopg.rows import TupleRow
from pydantic import ConfigDict

from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import Record

type Block = BetaTextBlockParam | BetaSearchResultBlockParam


@dataclass(frozen=True)
class Desk:
    """One run of the agent: a connection already set to `person`, and the day it's asked on.

    Every tool reads through `conn`, so row-level security decides what it sees (rule 3), and
    stops at `as_of` (rule 2): a date Claude asks for past it is pulled back to it. `shown` and
    `read` are the chunks and whole emails this run has already handed Claude, so search sends
    new ones instead (a fifth to a quarter of results were repeats, 2026-11-10).
    """

    conn: psycopg.Connection[TupleRow]
    person: str
    as_of: date
    shown: set[int] = field(default_factory=set)
    read: set[int] = field(default_factory=set)

    @property
    def investigator(self) -> bool:
        """Whether this person may read everyone's mail and the radar."""
        return self.person == db.INVESTIGATOR

    def until(self, day: date | None) -> date:
        """The last day a tool reads: the day asked for, never past `as_of`."""
        return self.as_of if day is None else min(day, self.as_of)


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns: blocks for Claude, one line for the person watching, its documents.

    `aside` is text for Claude that can't sit in the result itself (a result of search results
    takes nothing else): the loop sends it right after.
    """

    blocks: Sequence[Block]
    summary: str
    documents: tuple[int, ...] = ()
    aside: str = ""

    def __post_init__(self) -> None:
        # The API refuses a tool result mixing search results with anything else (400, found on
        # the first smoke run, 2026-11-10): caught here, in tests, instead.
        kinds = {block["type"] for block in self.blocks}
        if "search_result" in kinds and len(kinds) > 1:
            raise ValueError(f"a tool result mixes {sorted(kinds)}")


class ToolInput(Record):
    """A tool's arguments as Claude sends them; anything not declared is refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def text(body: str) -> BetaTextBlockParam:
    """Plain text for Claude to read."""
    return {"type": "text", "text": body}


def local_day(instant: datetime) -> date:
    """The store's local day an instant falls on (America/Chicago)."""
    return instant.astimezone(ZoneInfo(get_settings().timezone)).date()


def week_of(day: date) -> date:
    """The Sunday a day's week ends on: the radar names its weeks by their last day."""
    return day + timedelta(days=6 - day.weekday())
