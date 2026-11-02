"""Keep press and bulk mail from filling the results, so what people wrote inside gets seen."""

import re
from collections.abc import Iterable
from enum import StrEnum

from rag_trend_radar.schema import SearchHit

## An investigator's top 10 was 89% press on story questions (107 of 120, 2026-11-02), with press
## about 6% of the chunks. At most this many of a list are press or bulk mail, unless nothing
## else is left.
RESULTS = 10
MAX_PRESS = 3
MAX_BULK = 1
## A rerank score under this is noise ("UNBELIEVABLE", "FYI"): such a chunk never takes the place
## of a press article.
MIN_SCORE = 0.05

_PRESS_SUBJECT = re.compile(
    r"enron mentions|energy issues|iep news|industry mentions|media coverage|daily gas news"
    r"|energy market report|^articles?\b|rigzone|enerfax|williams energy news|nytimes\.com"
    r"|article from|e-journal|hot today|issuealert|the world this week|headlines",
    re.IGNORECASE,
)
## A forwarded article carries its wire's mark. "Bloomberg" alone won't do: traders' own mail
## names the terminal.
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


class Source(StrEnum):
    """Who a chunk's words came from, as far as its own text says."""

    MAIL = "mail"
    PRESS = "press"
    BULK = "bulk"


def source(text: str) -> Source:
    """A chunk's source, from the subject and sender that lead it and the wire marks in it."""
    lead, _, body = text.partition("\n\n")
    subject = re.search(r"^Subject: (.*)$", lead, re.MULTILINE)
    sender = re.search(r"^From: (.*)$", lead, re.MULTILINE)
    title = subject[1] if subject else ""
    if _BULK_SUBJECT.search(title) or (sender and _BULK_SENDER.search(sender[1])):
        return Source.BULK
    if _PRESS_SUBJECT.search(title) or _PRESS_BODY.search(body):
        return Source.PRESS
    return Source.MAIL


def mix(hits: Iterable[SearchHit], top: int) -> list[SearchHit]:
    """The best `top` of a best-first list under the caps, best first.

    What the caps hold back comes back only when nothing else is left, ahead of mail scoring
    under `MIN_SCORE`.
    """
    limits = {Source.MAIL: top, Source.PRESS: MAX_PRESS, Source.BULK: MAX_BULK}
    taken = dict.fromkeys(Source, 0)
    chosen: list[SearchHit] = []
    held: list[SearchHit] = []
    noise: list[SearchHit] = []
    for hit in hits:
        kind = source(hit.text)
        if kind is Source.MAIL and hit.score < MIN_SCORE:
            noise.append(hit)
        elif taken[kind] < limits[kind]:
            taken[kind] += 1
            chosen.append(hit)
        else:
            held.append(hit)
    chosen += (held + noise)[: max(top - len(chosen), 0)]
    return sorted(chosen[:top], key=lambda hit: -hit.score)
