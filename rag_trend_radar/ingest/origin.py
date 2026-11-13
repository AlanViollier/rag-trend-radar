"""Where a chunk's words came from: people's own mail, the press, or bulk mail."""

import re
from enum import StrEnum

_PRESS_SUBJECT = re.compile(
    r"enron mentions|energy issues|iep news|industry mentions|media coverage|daily gas news"
    r"|energy market report|^articles?\b|rigzone|enerfax|williams energy news|nytimes\.com"
    r"|article from|e-journal|hot today|issuealert|the world this week|headlines",
    re.IGNORECASE,
)
# A forwarded article carries its wire's mark. "Bloomberg" alone won't do: traders' own mail
# names the terminal.
_PRESS_BODY = re.compile(
    r"\(Dow Jones\)|\(AP\)|\(Bloomberg\)|\(Reuters\)|Associated Press|Copyright (\(c\)|\?|©)"
)
_BULK_SUBJECT = re.compile(
    r"commissioner\.com|e-reports|real deals|net saaver|news alert|schedule crawler"
    r"|demand ken lay donate",
    re.IGNORECASE,
)
_BULK_SENDER = re.compile(
    r"commissioner\.com|nytdirect|travelocity|noreply|no-reply", re.IGNORECASE
)


class Origin(StrEnum):
    """Where a chunk's words came from, as far as its own text says."""

    MAIL = "mail"
    PRESS = "press"
    BULK = "bulk"


def origin(text: str) -> Origin:
    """A chunk's origin, from the subject and sender that lead it and the wire marks in it."""
    lead, _, body = text.partition("\n\n")
    subject = re.search(r"^Subject: (.*)$", lead, re.MULTILINE)
    sender = re.search(r"^From: (.*)$", lead, re.MULTILINE)
    title = subject[1] if subject else ""
    if _BULK_SUBJECT.search(title) or (sender and _BULK_SENDER.search(sender[1])):
        return Origin.BULK
    if _PRESS_SUBJECT.search(title) or _PRESS_BODY.search(body):
        return Origin.PRESS
    return Origin.MAIL
