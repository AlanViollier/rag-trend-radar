"""Enron's mail: which blobs are newsletters, footers or chatter rather than a subject.

The one file here that knows this corpus. `blobs.kind` is judged over a blob's whole life, with
hindsight; an alert judges a blob on its weeks so far. The naming pass's reading replaces this on
the page, never for alerts: what's read with hindsight can't decide what fires.
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

_SENDER = re.compile(r"From:\s*(.+?)(?:\s{2,}|\s+Date:|\n|$)")
_SUBJECT = re.compile(r"Subject:\s*(.+?)(?:\s+From:|\n|$)", re.IGNORECASE)
_REPLY = re.compile(r"^(re|fw|fwd):\s*")
_FOOTER = re.compile(
    r"intended (only )?(for|recipient)|privileged|\bfax\b|entry description|calendar entry"
    r"|forwarded by|stdolelink",
    re.IGNORECASE,
)
_CHATTY_WORDS = (
    "i'm going just know good thanks don't did he's think weekend night time let want got like"
    " fyi fw fwd doc re cc hi hey see great really jpg mvc- exe gif mpg mpeg click unsubscribe"
    " remove http html www hotmail aol yahoo com"
)
_CHATTY = set(_CHATTY_WORDS.split())
ONE_SENDER, EMPTY_SUBJECTS = 0.8, 0.15
# Set on a blob's top 6 keywords; a blob-week stores 10.
CHATTY_KEYWORDS = 6


class Kind(StrEnum):
    """What a blob is. Only a topic goes on the map."""

    TOPIC = "topic"
    NEWSLETTER = "newsletter"
    FOOTER = "footer"
    CHATTER = "chatter"


@dataclass(frozen=True)
class Mail:
    """Each pool row's sender and subject, and whether it holds a footer."""

    senders: list[str]
    subjects: list[str]
    footers: NDArray[np.bool_]


@dataclass(frozen=True)
class Trait:
    """A blob in one week, for judging it."""

    chunks: int
    top_sender: float
    footers: float
    empty_subjects: int
    chatty_words: int


def read(texts: Sequence[str]) -> Mail:
    """The header fields and footers of every chunk."""
    return Mail(
        [_field(_SENDER, text) for text in texts],
        [_REPLY.sub("", _field(_SUBJECT, text)) for text in texts],
        np.array([bool(_FOOTER.search(text)) for text in texts], dtype=bool),
    )


def trait(mail: Mail, rows: NDArray[np.int64], words: Sequence[str]) -> Trait:
    """A blob's week, from the pool rows it held and its keywords."""
    picked = rows.tolist()
    return Trait(
        chunks=len(picked),
        top_sender=Counter(mail.senders[i] for i in picked).most_common(1)[0][1] / len(picked),
        footers=float(mail.footers[rows].mean()),
        empty_subjects=sum(not mail.subjects[i] for i in picked),
        chatty_words=sum(word in _CHATTY for word in words[:CHATTY_KEYWORDS]),
    )


def kind(weeks: Sequence[Trait]) -> Kind:
    """A blob's kind from its whole life: one sender, footers, or mail with nothing to say."""
    if np.median([w.top_sender for w in weeks]) >= ONE_SENDER:
        return Kind.NEWSLETTER
    footer = [w.footers >= 0.5 for w in weeks]
    chatty = [w.chatty_words >= 2 for w in weeks]
    empty = sum(w.empty_subjects for w in weeks) / sum(w.chunks for w in weeks)
    if np.mean(footer) >= 0.5:
        return Kind.FOOTER
    if (
        empty >= EMPTY_SUBJECTS
        or np.mean([f or c for f, c in zip(footer, chatty, strict=True)]) >= 0.5
    ):
        return Kind.CHATTER
    return Kind.TOPIC


def _field(pattern: re.Pattern[str], text: str) -> str:
    found = pattern.search(text)
    return found[1].strip().lower() if found else ""
