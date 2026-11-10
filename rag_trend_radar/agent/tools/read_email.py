"""One email whole, with who it went to and what it quotes."""

from psycopg import sql
from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, local_day, text
from rag_trend_radar.answer import cited
from rag_trend_radar.days import cutoff
from rag_trend_radar.search.ranking import SEEN

DESCRIPTION = """Read one email whole: sender, recipients, date, its full text and what it \
quotes. Use it when a search result is cut \
short or you need who it went to. Citable like a search result."""
# Enough for a long memo; past it the email is cut and says so.
MAX_BODY, MAX_QUOTED = 12_000, 4_000
SHOWN_RECIPIENTS = 12

EMAIL = sql.SQL(
    "SELECT d.kind, d.sent_at, d.title, d.body, d.quoted, e.sender, e.to_addrs, e.cc_addrs"
    " FROM documents d LEFT JOIN emails e ON e.document_id = d.id"
    " WHERE d.id = %(id)s AND {seen}"
).format(seen=SEEN)


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    document_id: int = Field(description="The document id a search result or count named.")


def run(desk: Desk, given: Input) -> ToolResult:
    """The email as one citable block, headers first."""
    params = {"id": given.document_id, "cutoff": cutoff(desk.as_of)}
    rows = desk.conn.execute(EMAIL, params).fetchall()
    if not rows:
        return ToolResult(
            [text(f"No document {given.document_id} this person could read by {desk.as_of}.")],
            "not readable",
        )
    kind, sent_at, title, body, quoted, sender, to, cc = rows[0]
    header = [
        f"Subject: {title}",
        f"From: {sender or 'unknown (a message known only from replies quoting it)'}",
        f"To: {_some(to or [])}",
        f"Cc: {_some(cc or [])}",
        f"Date: {local_day(sent_at)}",
    ]
    full = "\n".join(header) + "\n\n" + _cut(body, MAX_BODY)
    notes = f"document {given.document_id} ({kind})"
    return ToolResult(
        [text(notes), cited(given.document_id, full, _cut(quoted, MAX_QUOTED))],
        f"read document {given.document_id}, {local_day(sent_at)}",
        (given.document_id,),
    )


def _some(addresses: list[str]) -> str:
    shown = ", ".join(addresses[:SHOWN_RECIPIENTS])
    more = len(addresses) - SHOWN_RECIPIENTS
    return f"{shown} and {more} more" if more > 0 else shown or "nobody listed"


def _cut(body: str, limit: int) -> str:
    return (
        body
        if len(body) <= limit
        else body[:limit] + f"\n\n[cut: {len(body) - limit} more characters]"
    )
