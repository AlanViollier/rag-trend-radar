"""The radar's topics that grew most, or were biggest, over a stretch of weeks."""

from collections import defaultdict
from datetime import date, timedelta
from typing import Literal, NamedTuple

from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, text
from rag_trend_radar.days import weeks_within
from rag_trend_radar.radar import naming

DESCRIPTION = """The radar's topics (clusters of similar mail, followed week by week) that grew \
most or were biggest between two days. Use it for what was rising, what people were talking \
about, or to find a topic's id by a word. "Recent mail" is the emails in or near the topic in \
the 4 weeks up to a week. Newsletters, footers and chatter are left out. Then topic_history for \
one topic, or search for its mail. Investigator only."""
MAX_TOPICS = 25
DEFAULT_SPAN = timedelta(weeks=4)
LOOK_BACK = timedelta(weeks=8)

ROWS = (
    "SELECT bw.cluster_id, b.kind, bw.week, bw.recent, bw.mail, bw.keywords FROM cluster_weeks bw"
    " JOIN clusters b ON b.id = bw.cluster_id"
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
    topic missing that week (clusters skip weeks) takes its last week before it, up to `LOOK_BACK`.
    """
    until = desk.until(given.until)
    first, last = weeks_within(given.since or until - DEFAULT_SPAN, until)
    before = first - timedelta(weeks=1)
    rows = desk.conn.execute(ROWS, (before - LOOK_BACK, last)).fetchall()
    if not rows:
        return ToolResult([text(f"No radar weeks between {first} and {last}.")], "no weeks")
    weeks: dict[int, dict[date, tuple[int, int, list[str]]]] = defaultdict(dict)
    baseline: dict[int, int] = {}
    for cluster, built, week, recent, mail, keywords in rows:
        if naming.kind(cluster, built) != "topic":
            continue
        if week <= before:
            baseline[cluster] = recent  # rows come oldest first: the last one before the range wins
        else:
            weeks[cluster][week] = (recent, mail, keywords)
    alerts: dict[int, int] = dict(
        desk.conn.execute(
            "SELECT cluster_id, count(*) FROM alerts WHERE day BETWEEN %s AND %s GROUP BY cluster_id",
            (first, last),
        ).fetchall()
    )
    found: list[_Topic] = []
    for cluster, seen in weeks.items():
        latest = max(seen)
        label = naming.name(cluster, latest, seen[latest][2])
        wanted = f"{label} {' '.join(w for row in seen.values() for w in row[2])}".lower()
        if given.words and not all(word in wanted for word in given.words.lower().split()):
            continue
        start, end = baseline.get(cluster, 0), (seen[last][0] if last in seen else 0)
        size = sum(mail for _, mail, _ in seen.values()) / len(seen)
        peak = max(recent for recent, _, _ in seen.values())
        found.append(_Topic(cluster, label, start, end, peak, size, seen[latest][2]))
    if given.order == "growth":
        found.sort(key=lambda topic: topic.end - topic.start, reverse=True)
    else:
        found.sort(key=lambda topic: topic.size, reverse=True)
    lines = [
        f"Topics, weeks ending {first} to {last}, by {given.order} since week ending {before}:"
    ] + [
        f"topic {t.cluster} · {t.label} · recent mail {t.start} → {t.end} ({t.end - t.start:+})"
        f" · peak {t.peak} · {t.size:.0f} mail a week · {alerts.get(t.cluster, 0)} alerts"
        f" · keywords: {', '.join(t.keywords[:6])}"
        for t in found[: min(given.limit, MAX_TOPICS)]
    ]
    if not found:
        lines.append("none match")
    return ToolResult([text("\n".join(lines))], f"{len(found)} topics, {first} to {last}")


class _Topic(NamedTuple):
    cluster: int
    label: str
    start: int
    end: int
    peak: int
    size: float
    keywords: list[str]
