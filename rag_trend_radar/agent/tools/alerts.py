"""The radar's alerts between two days: topics that suddenly grew far past their normal."""

from datetime import date, timedelta

from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, text
from rag_trend_radar.agent.tools import topics

DESCRIPTION = """The radar's alerts between two days. An alert fires when a topic's last 4 weeks \
hold 4 times its own normal and 1% of all the mail the radar reads; about one every 9 days. Each \
names its topic id for topic_history. Newsletters and other junk are marked. Investigator only."""
MAX_ALERTS = 60
DEFAULT_SPAN = timedelta(weeks=8)

ALERTS = (
    "SELECT a.day, a.blob_id, b.kind, a.observed, a.expected, (SELECT bw.keywords FROM blob_weeks bw"
    " WHERE bw.blob_id = a.blob_id AND bw.week <= a.day ORDER BY bw.week DESC LIMIT 1)"
    " FROM alerts a JOIN blobs b ON b.id = a.blob_id"
    " WHERE a.day BETWEEN %s AND %s ORDER BY a.day LIMIT %s"
)


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    since: date | None = Field(None, description="First day; default 8 weeks before `until`.")
    until: date | None = Field(None, description="Last day; default today.")


def run(desk: Desk, given: Input) -> ToolResult:
    """One line per alert, oldest first, with the topic's name on that day."""
    until = desk.until(given.until)
    since = given.since or until - DEFAULT_SPAN
    rows = desk.conn.execute(ALERTS, (since, until, MAX_ALERTS)).fetchall()
    lines = [f"Alerts from {since} to {until}:"] + [
        f"{day} · topic {blob} · {topics.name(blob, day, keywords or [])}{_unless_topic(blob, built)}"
        f" · {observed / expected:.1f} times its normal"
        f" ({observed:.0f} chunks in 4 weeks, normal {expected:.0f})"
        for day, blob, built, observed, expected, keywords in rows
    ]
    if not rows:
        lines.append("none")
    return ToolResult([text("\n".join(lines))], f"{len(rows)} alerts, {since} to {until}")


def _unless_topic(blob: int, built: str) -> str:
    sort = topics.kind(blob, built)
    return "" if sort == "topic" else f" ({sort})"
