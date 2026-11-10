"""The radar's topics that grew most, or were biggest, over a stretch of weeks."""

from collections import defaultdict
from datetime import date, timedelta
from typing import Literal

from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, text
from rag_trend_radar.agent.tools import topics

DESCRIPTION = """The radar's topics (clusters of similar mail, followed week by week) that grew \
most or were biggest between two days. Use it for what was rising, what people were talking \
about, or to find a topic's id by a word. "Recent mail" is the emails in or near the topic in \
the 4 weeks up to a week. Newsletters, footers and chatter are left out. Then topic_history for \
one topic, or search for its mail. Investigator only."""
MAX_TOPICS = 25
DEFAULT_SPAN = timedelta(weeks=4)
LOOK_BACK = timedelta(weeks=8)

ROWS = (
    "SELECT bw.blob_id, b.kind, bw.week, bw.recent, bw.mail, bw.keywords FROM blob_weeks bw"
    " JOIN blobs b ON b.id = bw.blob_id"
    " WHERE b.kind = 'topic' AND b.shown AND bw.week BETWEEN %s AND %s ORDER BY bw.week"
)


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    since: date | None = Field(None, description="First day; default 4 weeks before `until`.")
    until: date | None = Field(None, description="Last day; default today.")
    order: Literal["growth", "size"] = Field(
        "growth", description='"growth": most recent mail gained; "size": most mail on average.'
    )
    words: str | None = Field(
        None, description="Only topics whose name or keywords hold these words."
    )
    limit: int = Field(10, description=f"How many topics, at most {MAX_TOPICS}.")


def run(desk: Desk, given: Input) -> ToolResult:
    """Each shown topic's recent mail the week before the range and at its last week, ranked.

    Growth counts from the week before the first: a one-week range measured from its own first
    week gave every topic +0 and hid the Dynegy merger the week it fell apart (2026-11-10). A
    topic missing that week (blobs skip weeks) takes its last week before it, up to `LOOK_BACK`.
    """
    until = desk.until(given.until)
    first, last = topics.weeks(given.since or until - DEFAULT_SPAN, until)
    before = first - timedelta(weeks=1)
    rows = desk.conn.execute(ROWS, (before - LOOK_BACK, last)).fetchall()
    if not rows:
        return ToolResult([text(f"No radar weeks between {first} and {last}.")], "no weeks")
    weeks: dict[int, dict[date, tuple[int, int, list[str]]]] = defaultdict(dict)
    baseline: dict[int, int] = {}
    for blob, built, week, recent, mail, keywords in rows:
        if topics.kind(blob, built) != "topic":
            continue
        if week <= before:
            baseline[blob] = recent  # rows come oldest first: the last one before the range wins
        else:
            weeks[blob][week] = (recent, mail, keywords)
    alerts: dict[int, int] = dict(
        desk.conn.execute(
            "SELECT blob_id, count(*) FROM alerts WHERE day BETWEEN %s AND %s GROUP BY blob_id",
            (first, last),
        ).fetchall()
    )
    found = []
    for blob, seen in weeks.items():
        latest = max(seen)
        label = topics.name(blob, latest, seen[latest][2])
        wanted = f"{label} {' '.join(w for row in seen.values() for w in row[2])}".lower()
        if given.words and not all(word in wanted for word in given.words.lower().split()):
            continue
        start, end = baseline.get(blob, 0), (seen[last][0] if last in seen else 0)
        size = sum(mail for _, mail, _ in seen.values()) / len(seen)
        found.append(
            (blob, label, start, end, max(r for r, _, _ in seen.values()), size, seen[latest][2])
        )
    key = (lambda t: t[3] - t[2]) if given.order == "growth" else (lambda t: t[5])
    found.sort(key=key, reverse=True)
    lines = [
        f"Topics, weeks ending {first} to {last}, by {given.order} since week ending {before}:"
    ] + [
        f"topic {blob} · {label} · recent mail {start} → {end} ({end - start:+}) · peak {peak}"
        f" · {size:.0f} mail a week · {alerts.get(blob, 0)} alerts · keywords: {', '.join(kw[:6])}"
        for blob, label, start, end, peak, size, kw in found[: min(given.limit, MAX_TOPICS)]
    ]
    if not found:
        lines.append("none match")
    return ToolResult([text("\n".join(lines))], f"{len(found)} topics, {first} to {last}")
