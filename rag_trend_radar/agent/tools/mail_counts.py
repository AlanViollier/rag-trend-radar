"""How much mail matches a phrase, senders, recipients and dates, by week, month or sender.

Fixed SQL with Claude's values as parameters, never SQL Claude writes (STATE, 2026-11-09).
"""

from collections import Counter
from datetime import date, datetime
from typing import Literal

from psycopg import sql
from pydantic import Field, field_validator

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, local_day, text, week_of
from rag_trend_radar.days import cutoff, opening
from rag_trend_radar.ingest.kinds import Source, source

DESCRIPTION = """Count emails (not forwards of the same one) by week, month or sender, filtered by \
a phrase, senders, recipients, source and dates. Use it for when something started or peaked, \
who wrote about it most, or who wrote to whom. Weeks are named by their last day, a Sunday, as \
on the radar. Get addresses from find_person first. Counts people's emails, each once."""
# Telling mail from press reads each email's first chunk; past this many the phrase or dates
# must narrow it first.
MAX_SOURCED = 20_000
MAX_WEEKS, MAX_SENDERS = 120, 15

BASE = sql.SQL(
    "SELECT d.sent_at, e.sender, {lead} FROM documents d JOIN emails e ON e.document_id = d.id"
    " WHERE d.kind = 'email' AND d.sent_at >= %(since)s AND d.sent_at < %(cutoff)s"
)
LEAD = sql.SQL(
    "(SELECT c.text FROM chunks c WHERE c.document_id = d.id ORDER BY c.position LIMIT 1)"
)
# `###` is pg_search's phrase match, on the same tokens the BM25 index keeps.
PHRASE = sql.SQL(" AND d.id IN (SELECT c.document_id FROM chunks c WHERE c.text ### %(words)s)")
SENDERS = sql.SQL(" AND lower(e.sender) = ANY(%(senders)s)")
RECIPIENTS = sql.SQL(
    " AND (e.to_addrs && %(recipients)s OR e.cc_addrs && %(recipients)s"
    " OR e.bcc_addrs && %(recipients)s)"
)


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    words: str | None = Field(None, description="A word or phrase the email must contain.")
    senders: list[str] = Field([], description="Sent from any of these addresses.")
    recipients: list[str] = Field([], description="Sent to, cc or bcc any of these addresses.")
    source: Literal["any", "mail", "press", "bulk"] = Field(
        "any",
        description='"mail": people\'s own writing; "press": forwarded articles; "bulk": newsletters, ads.',
    )
    since: date | None = Field(None, description="Only mail sent on or after this day.")
    until: date | None = Field(None, description="Only mail sent on or before this day.")
    by: Literal["week", "month", "sender", "total"] = Field("week", description="How to group.")

    @field_validator("senders", "recipients", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        # Claude sometimes sends "a@x.com,b@x.com" for a list (smoke run 2026-11-10): refused,
        # it didn't retry, and a count its answer needed was never made.
        return (
            [part.strip() for part in value.split(",") if part.strip()]
            if isinstance(value, str)
            else value
        )


def run(desk: Desk, given: Input) -> ToolResult:
    """The matching emails counted in Python, after one query fetches when and who."""
    query = BASE.format(lead=LEAD if given.source != "any" else sql.SQL("NULL"))
    if given.words:
        query += PHRASE
    if given.senders:
        query += SENDERS
    if given.recipients:
        query += RECIPIENTS
    if given.source != "any":
        query += sql.SQL(" LIMIT {}").format(MAX_SOURCED + 1)
    params = {
        "since": opening(given.since),
        "cutoff": cutoff(desk.until(given.until)),
        "words": given.words,
        "senders": [a.lower() for a in given.senders],
        "recipients": [a.lower() for a in given.recipients],
    }
    rows: list[tuple[datetime, str, str | None]] = desk.conn.execute(query, params).fetchall()
    if given.source != "any":
        if len(rows) > MAX_SOURCED:
            return ToolResult(
                [
                    text(
                        f"Over {MAX_SOURCED:,} emails: narrow by words, senders or dates to sort by source."
                    )
                ],
                "too broad",
            )
        rows = [row for row in rows if source(row[2] or "") is Source(given.source)]
    if not rows:
        return ToolResult([text("No emails match.")], "0 emails")
    days = sorted(local_day(sent) for sent, _, _ in rows)
    lines = [f"{len(rows):,} emails match. First {days[0]}, last {days[-1]}."]
    lines += _grouped(given.by, days, [sender for _, sender, _ in rows])
    return ToolResult([text("\n".join(lines))], f"{len(rows):,} emails, {days[0]} to {days[-1]}")


def _grouped(by: str, days: list[date], senders: list[str]) -> list[str]:
    if by == "sender":
        top = Counter(senders).most_common(MAX_SENDERS)
        return ["Top senders:"] + [f"{s}  {n}" for s, n in top]
    weeks = Counter(week_of(day) for day in days)
    if by == "week" and len(weeks) <= MAX_WEEKS:
        return ["Week ending  emails"] + [f"{w}  {n}" for w, n in sorted(weeks.items())]
    if by == "total":
        return []
    months = Counter(day.strftime("%Y-%m") for day in days)
    head = "Month  emails" if by == "month" else f"Over {MAX_WEEKS} weeks, by month instead:"
    return [head] + [f"{m}  {n}" for m, n in sorted(months.items())]
