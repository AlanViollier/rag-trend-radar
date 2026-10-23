"""One document per distinct email, however many mailboxes hold a copy."""

import hashlib
from collections.abc import Iterable
from datetime import datetime


def content_hash(subject: str, body: str) -> str:
    """Fingerprint equal for every copy of one email.

    Sender and date are left out on purpose: the Lotus and Outlook exports rewrite senders
    (`walton@` vs `leann.walton@`) and shift dates by whole hours, while subject and body survive
    both up to whitespace.
    """
    text = subject.strip().lower() + "\x00" + " ".join(body.split())
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


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
