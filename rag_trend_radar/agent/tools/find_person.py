"""A name to the mailbox and the addresses a person wrote from."""

import re

from psycopg import sql
from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, text
from rag_trend_radar.days import cutoff, local_day

DESCRIPTION = """Find a person by name: their mailbox, if they're one of the 150 employees whose \
mail is in the store, and every address they sent from, with how much mail and when. People \
wrote under several addresses and assistants sent for their boss, so pass the addresses you \
choose to mail_counts."""
MAX_ADDRESSES = 12
# "lay" is inside nicolay and clayton: the most-used matches are fetched, then ranked by how
# the address holds the name.
CANDIDATES = 60

SENDERS = sql.SQL(
    "SELECT e.sender, count(*), min(d.sent_at), max(d.sent_at)"
    " FROM emails e JOIN documents d ON d.id = e.document_id"
    " WHERE e.sender ILIKE %(pattern)s AND d.sent_at < %(cutoff)s"
    " GROUP BY e.sender ORDER BY count(*) DESC LIMIT %(limit)s"
)


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    name: str = Field(description='A name as people wrote it: "Vince Kaminski", "Lay", "Fastow".')


def run(desk: Desk, given: Input) -> ToolResult:
    """Mailboxes and sender addresses holding the name's last word, most mail first."""
    words = re.findall(r"[a-z0-9]+", given.name.lower())
    if not words or len(words[-1]) < 2:
        return ToolResult([text("Give at least a last name.")], "no name")
    # The last name alone: first names come as initials, nicknames or not at all (klay@).
    surname = words[-1]
    mailboxes = [
        m
        for (m,) in desk.conn.execute(
            "SELECT mailbox FROM employees WHERE mailbox LIKE %s ORDER BY mailbox",
            (f"{surname}-%",),
        ).fetchall()
    ]
    senders = desk.conn.execute(
        SENDERS,
        {"pattern": f"%{surname}%", "cutoff": cutoff(desk.as_of), "limit": CANDIDATES},
    ).fetchall()
    senders = sorted(senders, key=lambda row: (-_fit(row[0], words), -row[1]))[:MAX_ADDRESSES]
    lines = [f"Mailboxes: {', '.join(mailboxes) or 'none: not one of the 150 employees'}"]
    lines += ["Addresses that sent mail (emails, first, last):"] + [
        f"{address}  {n}  {local_day(first)}  {local_day(last)}"
        for address, n, first, last in senders
    ]
    if not senders:
        lines.append("none")
    summary = f"{len(mailboxes)} mailboxes, {len(senders)} addresses for {surname!r}"
    return ToolResult([text("\n".join(lines))], summary)


def _fit(address: str, words: list[str]) -> int:
    # 2 when the last name stands alone (kenneth.lay, "Lay, Kenneth"), 1 more for an initial
    # glued to it (klay) or the first name present.
    *given, surname = words
    local = address.lower().partition("@")[0]
    whole = re.search(rf"(^|[^a-z]){surname}($|[^a-z])", local) is not None
    initial = re.fullmatch(rf"[a-z]\.?{surname}[0-9]*", local) is not None
    first = bool(given) and given[0] in local
    return 2 * whole + initial + first
