"""One radar topic week by week: its size, words, alerts and who wrote in it."""

from datetime import date

from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, text
from rag_trend_radar.agent.tools import topics
from rag_trend_radar.days import cutoff, opening

DESCRIPTION = """One radar topic week by week: emails in or near it, recent mail (the last 4 \
weeks) and how much of that was press, chunks above its normal, its keywords when they \
change, its alerts, and the people who wrote the most in it. Take the id from top_topics or \
alerts. Investigator only."""
MAX_SENDERS = 10

CLUSTER = "SELECT kind, first_week, last_week FROM clusters WHERE id = %s"
WEEKS = (
    "SELECT week, mail, recent, recent_press, surge, keywords FROM cluster_weeks"
    " WHERE cluster_id = %s AND week BETWEEN %s AND %s ORDER BY week"
)
SENDERS = (
    "SELECT e.sender, count(DISTINCT d.id) FROM cluster_chunks bc JOIN chunks c ON c.id = bc.chunk_id"
    " JOIN documents d ON d.id = c.document_id JOIN emails e ON e.document_id = d.id"
    " WHERE bc.cluster_id = %s AND d.sent_at >= %s AND d.sent_at < %s"
    " GROUP BY e.sender ORDER BY 2 DESC LIMIT %s"
)


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    topic: int = Field(description="The topic's id.")
    since: date | None = Field(None, description="First day; default its first week.")
    until: date | None = Field(None, description="Last day; default today.")


def run(desk: Desk, given: Input) -> ToolResult:
    """Its weeks in range as lines, keywords only where they changed."""
    cluster = desk.conn.execute(CLUSTER, (given.topic,)).fetchall()
    if not cluster:
        return ToolResult([text(f"No topic {given.topic}.")], "no topic")
    kind, first_week, _ = cluster[0]
    until = desk.until(given.until)
    first, last = topics.weeks(given.since or first_week, until)
    rows = desk.conn.execute(WEEKS, (given.topic, first, last)).fetchall()
    if not rows:
        return ToolResult(
            [text(f"Topic {given.topic} has no weeks from {first} to {last}.")], "no weeks"
        )
    names = topics.every_name(given.topic, last)
    lines = [
        f"topic {given.topic} ({topics.kind(given.topic, kind)}), first seen week ending {first_week}"
    ]
    lines += [f"named from {start}: {label}" for start, label in names] or [
        "no name; keywords below"
    ]
    lines.append("Week ending  mail  recent (press)  chunks above its normal  keywords if changed")
    previous: list[str] = []
    for week, mail, recent, press, surge, keywords in rows:
        words = ", ".join(keywords[:6]) if keywords[:6] != previous else ""
        lines.append(f"{week}  {mail}  {recent} ({press})  {surge:+.0f}  {words}".rstrip())
        previous = keywords[:6]
    alerts = desk.conn.execute(
        "SELECT day, observed, expected FROM alerts WHERE cluster_id = %s AND day BETWEEN %s AND %s ORDER BY day",
        (given.topic, first, last),
    ).fetchall()
    lines += [
        f"alert {day}: {observed} chunks in 4 weeks against a normal of {expected:.0f}"
        for day, observed, expected in alerts
    ]
    senders = desk.conn.execute(
        SENDERS, (given.topic, opening(given.since or first_week), cutoff(until), MAX_SENDERS)
    ).fetchall()
    lines += ["Wrote the most in it:"] + [f"{sender}  {n}" for sender, n in senders]
    return ToolResult(
        [text("\n".join(lines))], f"topic {given.topic}: {len(rows)} weeks, {len(alerts)} alerts"
    )
