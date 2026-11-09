"""One document per distinct email, however many mailboxes hold a copy."""

import hashlib
import re
from collections.abc import Iterable
from datetime import datetime


def content_hash(subject: str, body: str) -> str:
    """Fingerprint equal for every copy of one email.

    Sender and date are left out on purpose: the Lotus and Outlook exports rewrite senders
    (`walton@` vs `leann.walton@`) and shift dates by whole hours, while subject and body survive
    both up to whitespace.
    """
    # The body is stripped before joining: one export starts it with a blank line, and a space
    # left after the separator would split the copies (756 emails, found 2026-10-29).
    return text_hash(subject.strip().lower() + "\x00" + body.strip())


def text_hash(text: str) -> str:
    """Fingerprint of `text` up to whitespace: a quoted copy of a message matches the original."""
    return hashlib.blake2b(" ".join(text.split()).encode(), digest_size=16).hexdigest()


def words(text: str) -> set[str]:
    """The distinct lowercase words of `text`, digits included: two texts compared as sets."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def numbers(text: str) -> set[str]:
    """The distinct runs of digits in `text`: where a template's copies differ."""
    return set(re.findall(r"[0-9]+", text))


def resolve_sent_at(dates: Iterable[datetime]) -> tuple[datetime, bool]:
    """Pick one date for copies of an email; the flag says the pick may be hours off.

    A date stamped to the second came from a mail server and wins. Otherwise disagreeing copies
    are export damage with no way to tell which is right: the earliest is kept, flagged.
    """
    distinct = sorted(set(dates))
    if len(distinct) == 1:
        return distinct[0], False
    stamped = [d for d in distinct if d.second]
    if len(stamped) == 1:
        return stamped[0], False
    return distinct[0], True
